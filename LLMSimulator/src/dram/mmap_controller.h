// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once

#include <memory>
#include <vector>

#include "base/type.h"
#include "common/assert.h"
#include "common/type.h"
#include "dram/memory_config.h"
#include "dram/memory_object.h"
#include "module/tensor.h"

namespace llm_system {

/// @brief 地址映射与分配器：为张量建立 MemoryObject，将线性地址拆为 cube/channel/bank/row/col。
class MMapController : public std::enable_shared_from_this<MMapController> {
 public:
  using Ptr = std::shared_ptr<MMapController>;

  /// @brief MMapController 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  static Ptr Create(MemoryConfig memory_config) {
    return MMapController::Ptr(new MMapController(memory_config));
  }

  /// @brief 按张量映射类型分配地址对象；当前分派仅处理 ALL_CHANNEL 分支。
  void setMemoryObject(Tensor::Ptr tensor);

  /// @brief 通过 shared_from_this 返回当前 MMapController 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Ptr getPtr() { return shared_from_this(); }
  /// @brief 以基址加 bundle_idx*granul 得到访问地址，按普通映射拆分并转换为 Ramulator 地址向量。
  Ramulator::AddrVec_t getAddrVec(addr address, long long bundle_idx);
  /// @brief 以基址和访问块索引计算地址，按 LOGIC 映射拆分并转换为 Ramulator 地址向量。
  Ramulator::AddrVec_t getAddrVecLOGIC(addr address, long long bundle_idx);

  /// @brief 返回地址映射和内存请求展开所使用的访问粒度（字节）。
  int getGranul() { return memory_config.granul; }
  /// @brief 按值返回内存几何配置。
  MemoryConfig getConfig(){ return memory_config; }

 private:
  addr start_addr_normal;
  addr start_addr_logic;

  /// @brief 保存内存几何参数，并把普通/LOGIC 两套分配游标初始化为 0。
  MMapController(MemoryConfig memory_config);

   // channel = pseudochannel in this function
   /// @brief 为普通张量建立内存对象；权重等推进分配游标，act/cache 当前复用起始地址 0，是简化地址模型。
   void setNormal(Tensor::Ptr tensor);   // allocate to all channel
   /// @brief 以 LOGIC 分配游标建立张量地址对象，再按张量字节数推进游标。
   void setLOGIC(Tensor::Ptr tensor);    // allocate to channel in a cube

   // not used
   /// @brief 单通道分配的旧接口；当前项目中仅保留声明，未找到定义。
   void setChannel(Tensor::Ptr tensor);  // allocate to one channel

   /// @brief 按粒度、cube、通道、rank、bankgroup、bank、列、行的顺序分解普通线性地址。
   AddrVec addrToVec(addr addr);
   
   /// @brief 按 LOGIC 的 cube/channel/bank/列/行交错规则分解地址；bank 位分成上下两部分。
   AddrVec addrToVecLOGIC(addr addr);

   /// @brief 按实现中的 LOGIC 字段组合规则把分层地址编码回线性字节地址。
   addr vecToAddrLOGIC(AddrVec addr_vec);

   MemoryConfig memory_config;
};

}  // namespace llm_system
