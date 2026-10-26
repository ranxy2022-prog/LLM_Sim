// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include "dram/mmap_controller.h"

#include "common/assert.h"
#include "common/type.h"

namespace llm_system {

/// @brief 保存内存几何参数，并把普通/LOGIC 两套分配游标初始化为 0。
MMapController::MMapController(MemoryConfig memory_config)
    : memory_config(memory_config) {
  start_addr_normal = 0;
  start_addr_logic = 0;
};

/// @brief 按张量映射类型分配地址对象；当前分派仅处理 ALL_CHANNEL 分支。
void MMapController::setMemoryObject(Tensor::Ptr tensor) {
  MMap map = tensor->getMMap();
  if (map == MMap::ALL_CHANNEL) {
    setNormal(tensor);
  }
}

/// @brief 以基址加 bundle_idx*granul 得到访问地址，按普通映射拆分并转换为 Ramulator 地址向量。
Ramulator::AddrVec_t MMapController::getAddrVec(addr address, long long bundle_idx) {
  addr target_address = address + bundle_idx * memory_config.granul;
  AddrVec addr_vec = addrToVec(target_address);
  int channel = memory_config.num_channel * addr_vec.cube + addr_vec.channel;
  Ramulator::AddrVec_t ramul_addr_vec = {
      channel / 2,   channel % 2,  addr_vec.rank, addr_vec.bankgroup,
      addr_vec.bank, addr_vec.row, addr_vec.col};

  return ramul_addr_vec;
}

/// @brief 以基址和访问块索引计算地址，按 LOGIC 映射拆分并转换为 Ramulator 地址向量。
Ramulator::AddrVec_t MMapController::getAddrVecLOGIC(addr address,
                                                     long long bundle_idx) {
  addr target_address = address + bundle_idx * memory_config.granul;
  AddrVec addr_vec = addrToVecLOGIC(target_address);
  int channel = memory_config.num_channel * addr_vec.cube + addr_vec.channel;
  Ramulator::AddrVec_t ramul_addr_vec = {
      channel / 2,   channel % 2,  addr_vec.rank, addr_vec.bankgroup,
      addr_vec.bank, addr_vec.row, addr_vec.col};

  return ramul_addr_vec;
}

/// @brief 按粒度、cube、通道、rank、bankgroup、bank、列、行的顺序分解普通线性地址。
AddrVec MMapController::addrToVec(addr address) {
  assertTrue(address >= 0, "Address is not valid");
  long long idx = 0;

  address /= memory_config.granul;

  int cube_idx = address % memory_config.num_cube;
  address /= memory_config.num_cube;

  int channel_idx = address % memory_config.num_channel;
  address /= memory_config.num_channel;

  int rank_idx = address % memory_config.num_rank;
  address /= memory_config.num_rank;

  int bg_idx = address % memory_config.num_bankgroup;
  address /= memory_config.num_bankgroup;

  int bank_idx = address % memory_config.num_bank;
  address /= memory_config.num_bank;

  int col_idx = address % memory_config.num_col;
  address /= memory_config.num_col;

  int row_idx = address;
  assertTrue(row_idx < memory_config.num_row, "Unvalid memory hasing");

  AddrVec addr = {cube_idx, channel_idx, rank_idx, bg_idx,
                  bank_idx, row_idx,     col_idx};
  return addr;
}

/// @brief 按 LOGIC 的 cube/channel/bank/列/行交错规则分解地址；bank 位分成上下两部分。
AddrVec MMapController::addrToVecLOGIC(addr address) {
  long long idx = 0;

  address /= memory_config.granul;

  int cube_idx = address % memory_config.num_logic_cube;
  address /= memory_config.num_logic_cube;

  int channel_idx = address % memory_config.num_channel;
  address /= memory_config.num_channel;

  // PIM memory set
  int bg_idx = address % memory_config.num_bankgroup;
  address /= memory_config.num_bankgroup;

  int bank_under_idx = address % 2;
  address /= 2;

  // PIM memory set

  int col_idx = address % memory_config.num_col;
  address /= memory_config.num_col;

  int row_idx = address % memory_config.num_row;
  address /= memory_config.num_row;

  int bank_upper_idx = address % 2;
  address /= 2;

  int rank_idx = address % memory_config.num_rank;

  int bank_idx = bank_upper_idx * 2 + bank_under_idx;

  assertTrue(rank_idx < memory_config.num_rank, "Unvalid memory hasing");

  AddrVec addr = {cube_idx, channel_idx, rank_idx, bg_idx,
                  bank_idx, row_idx,     col_idx};
  return addr;
}

/// @brief 按实现中的 LOGIC 字段组合规则把分层地址编码回线性字节地址。
addr MMapController::vecToAddrLOGIC(AddrVec addr_vec) {
  addr address = 0;
  int cube_idx = addr_vec.cube;
  int channel_idx = addr_vec.channel;
  int rank_idx = addr_vec.rank;
  int bankgroup_idx = addr_vec.bankgroup;
  int bank_idx = addr_vec.bank;
  int row_idx = addr_vec.row;
  int col_idx = addr_vec.col;

  int bank_upper_idx = bank_idx / 2;
  int bank_under_idx = bank_idx % 2;

  address += rank_idx;

  address *= 2;
  address += bank_upper_idx;

  address *= memory_config.num_row;
  address += row_idx;

  address *= memory_config.num_col;
  address += col_idx;

  address *= 2;
  address += bank_under_idx;

  address *= memory_config.num_bankgroup;
  address += bankgroup_idx;

  address *= memory_config.num_channel;
  address += channel_idx;

  address *= memory_config.num_cube;
  address += cube_idx;

  address *= memory_config.granul;

  return address;
}

// allocate tensor in all channel
/// @brief 为普通张量建立内存对象；权重等推进分配游标，act/cache 当前复用起始地址 0，是简化地址模型。
void MMapController::setNormal(Tensor::Ptr tensor) {
  addr start_address = start_addr_normal;
  if (tensor->tag.compare("act") && tensor->tag.compare("cache")) {
    start_addr_normal += tensor->getSize();
  } else {
    start_address = 0;
  }

  MemoryObject::Ptr memory_object = MemoryObject::Create(
      tensor->getMMap(), start_address, tensor->getSize(), getPtr());
  tensor->allocateMemoryObject(memory_object);
}

// allocate tensor in all channel
/// @brief 以 LOGIC 分配游标建立张量地址对象，再按张量字节数推进游标。
void MMapController::setLOGIC(Tensor::Ptr tensor) {
  MemoryObject::Ptr memory_object = MemoryObject::Create(
      tensor->getMMap(), start_addr_logic, tensor->getSize(), getPtr());
  tensor->allocateMemoryObject(memory_object);
  start_addr_logic += tensor->getSize();
}

}  // namespace llm_system
