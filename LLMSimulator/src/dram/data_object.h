// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
// 历史遗留实现，未加入 dram 目标；当前使用 MemoryObject，旧 Address 接口并不完整。
#pragma once

#include <memory>
#include <vector>

#include "common/assert.h"
#include "common/type.h"
#include "dram/dram_type.h"

namespace llm_system {

/**
 * @brief
 */

/// @brief 未纳入当前构建的旧地址对象；Address 等接口已与现行 MemoryObject 设计不一致，仅作历史参考。
class DataObject {
 public:
  using Ptr = std::shared_ptr<DataObject>;

  /// @brief DataObject 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  static Ptr Create(MMap mmap, MemoryObject) {
    return DataObject::Ptr(new DataObject(mmap, address));
  }
  /// @brief DataObject 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  static Ptr Create(const Address &addr) { return Ptr(new DataObject(addr)); }

  /// @brief 旧接口：用保存的地址信息创建另一个 DataObject，并非复制实际内存中的数据。
  Ptr DeepCopy() { return Create(addr); }

  // Enable moving, but not copying
  /// @brief DataObject 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  DataObject(DataObject &&) = default;
  /// @brief 使用编译器默认的成员赋值规则；智能指针复制共享所有权，移动成员按其类型规则转移状态。
  DataObject &operator=(DataObject &&) = default;

  /// @brief 返回旧 Address 描述的只读引用；该类型在现行自有代码中没有完整定义。
  const Address &GetAddress() const { return addr; }

  /// @brief 旧接口：按块偏移除以每行列组数计算行偏移；使用已过时的 addr_ 字段。
  const int GetRowAddr(int offset);
  /// @brief 旧接口：按块偏移对每行列组数取模计算列组偏移；使用已过时的 addr_ 字段。
  const int GetColAddr(int offset);
  /// @brief 使用默认析构规则释放 DataObject 的成员；智能指针成员按所有权规则释放引用。
  ~DataObject() = default;

 private:
  /// @brief DataObject 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  DataObject() = default;
  /// @brief 旧地址对象构造接口，保存映射或地址描述；当前实现与头文件不一致且未加入构建。
  DataObject(const Address &addr) : addr{addr} {}
  /// @brief 旧地址对象构造接口，保存映射或地址描述；当前实现与头文件不一致且未加入构建。
  DataObject(MMap mmap, Address address);
  MMap mmap;
  Address addr;
};

}  // namespace llm_system
