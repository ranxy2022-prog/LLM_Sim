// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include "dram/memory_object.h"

#include "common/assert.h"
#include "common/type.h"
#include "dram/mmap_controller.h"

namespace llm_system {

/// @brief 记录映射类型、基址、大小和映射器；访问块数按 size/granul 整数除法计算。
MemoryObject::MemoryObject(MMap mmap, addr address, long long size,
                           MMapController_Ptr mmap_controller)
    : mmap(mmap),
      address(address),
      size(size),
      mmap_controller(mmap_controller) {
  num_bundle = size / mmap_controller->getGranul();
}

/// @brief 按处理器类型选择普通或 LOGIC 地址映射，返回指定访问块的 DRAM 分层地址。
Ramulator::AddrVec_t MemoryObject::getAddrVec(long long bundle_idx,
                                              ProcessorType type) {
  Ramulator::AddrVec_t addrvec;
  if (type == ProcessorType::LOGIC) {
    addrvec = mmap_controller->getAddrVecLOGIC(address, bundle_idx);
  }
  else {
    addrvec = mmap_controller->getAddrVec(address, bundle_idx);
  }

  return addrvec;
}

/// @brief 更新逻辑字节数，并用整数除法重新计算访问块数；不足一个粒度的尾部不另增块。
void MemoryObject::setSize(long long size_) {
  size = size_;
  num_bundle = size / mmap_controller->getGranul();
}

}  // namespace llm_system
