// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once

#include <memory>
#include <vector>

#include "base/type.h"
#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/base.h"

namespace llm_system {

// input address vector for ramulator
/// @brief DRAM 分层地址字段：区分 cube、通道、rank、bankgroup、bank、行与列。
struct AddrVec {
  int cube;
  int channel;
  int rank;
  int bankgroup;
  int bank;
  int row;
  int col;
};

/// @brief MMapController 的前向声明：此处只声明类型，完整结构见对应头文件。
class MMapController;

/// @brief 张量对应的模拟地址区间：记录基址、字节数和访问块数，通过映射控制器生成 DRAM 地址。
class MemoryObject {
 public:
  using Ptr = std::shared_ptr<MemoryObject>;
  using MMapController_Ptr = std::shared_ptr<MMapController>;

  /// @brief MemoryObject 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(MMap mmap, addr address, long long size,
                                  MMapController_Ptr mmap_controller) {
    return Ptr(new MemoryObject(mmap, address, size, mmap_controller));
  };

  /// @brief 按处理器类型选择普通或 LOGIC 地址映射，返回指定访问块的 DRAM 分层地址。
  Ramulator::AddrVec_t getAddrVec(long long bundle_idx,
                                  ProcessorType type = ProcessorType::GPU);
  /// @brief 返回按访问粒度划分的块数，命令内核据此遍历访问。
  long getBundleSize() { return num_bundle; }

  /// @brief 更新逻辑字节数，并用整数除法重新计算访问块数；不足一个粒度的尾部不另增块。
  void setSize(long long _size);

 private:
  MMap mmap;
  addr address;
  long long size;
  long long num_bundle;

  MMapController_Ptr mmap_controller;

  /// @brief MemoryObject 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  MemoryObject() = default;
  /// @brief 记录映射类型、基址、大小和映射器；访问块数按 size/granul 整数除法计算。
  MemoryObject(MMap mmap, addr address, long long size,
               MMapController_Ptr mmap_controller);
};

}  // namespace llm_system
