// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_object.h"

/// @brief 显式实例化操作数映射容器：把 PIM 操作数角色映射到一组内存对象。
template class std::unordered_map<llm_system::PIMOperandType,
                                  std::vector<llm_system::MemoryObject::Ptr>>;

namespace llm_system {

/// @brief 逻辑内存请求：保存读写/GEMV 等类型，以及按用途分组的 MemoryObject 操作数。
class DRAMRequest : public std::enable_shared_from_this<DRAMRequest> {
 public:
  friend class DRAMInterface;
  using Ptr = std::shared_ptr<DRAMRequest>;

  using PIM_Operand =
      std::unordered_map<PIMOperandType, std::vector<MemoryObject::Ptr>>;

  /// @brief DRAMRequest 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(DRAMRequestType type) {
    return Ptr(new DRAMRequest(type));
  }

  /// @brief 通过 shared_from_this 返回当前 DRAMRequest 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Ptr GetPtr() {
    return shared_from_this();
  }

  /// @brief 按操作数角色把 MemoryObject 加入请求；同一角色可包含多个地址对象。
  void AddOperand(const MemoryObject::Ptr arg, PIMOperandType type) {
    if (operands_.find(type) == operands_.end()) {
      operands_.emplace(type, std::vector<MemoryObject::Ptr>{});
    }
    operands_.at(type).push_back(arg);
  }

  /// @brief 返回逻辑请求种类，供选择命令展开内核。
  DRAMRequestType GetType() const { return type_; }

  /// @brief 使用默认析构规则释放 DRAMRequest 的成员；智能指针成员按所有权规则释放引用。
  ~DRAMRequest() = default;

  PIM_Operand operands_;

 private:
  // Duration of DRAMRequests are not determined at the moment of construction
  /// @brief DRAMRequest 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  DRAMRequest() = default;
  /// @brief 保存逻辑内存请求类型；此时尚未生成具体命令，也没有请求耗时结果。
  DRAMRequest(DRAMRequestType type);

  DRAMRequestType type_;
};

}  // namespace llm_system
