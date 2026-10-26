// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include <dram/dram_interface.h>
#include <dram/pimkernel/pim_kernel.h>

namespace llm_system {
namespace PIM_KERNEL {

/// @brief 遍历操作数的访问块并生成写命令，按实现的地址筛选与处理器分支构造请求。
void Write_kernel(PIMRequest& pim_request, DRAMRequestType dramreq_type,
                  DRAMRequest::PIM_Operand& operand,
                  const PIMHWConfig pim_hw_config) {
  auto write_operand = get_operand(operand, PIMOperandType::kDRAM);
  assertTrue((write_operand.size() == 1), "only one request can be written");
  Ramulator::AddrVec_t addr_vec = {0, 0, 0, 0, 0, 0};

  // std::cout << "size: " << std::to_string(write_operand[0]->getBundleSize())
  //           << std::endl;
  for (auto opnd : write_operand) {
    for (int bundle_idx = 0; bundle_idx < opnd->getBundleSize(); bundle_idx++) {
      addr_vec = opnd->getAddrVec(bundle_idx, pim_hw_config.type);
      if (addr_vec.at(0) == 0) {
        pim_request.AddCommand(PIMCommand(PIMCommandType::kWrite,
                                          PIMOperandType::kDRAM, addr_vec,
                                          &pim_request, dramreq_type));
      }
    }
  }
}

}  // namespace PIM_KERNEL
}  // namespace llm_system