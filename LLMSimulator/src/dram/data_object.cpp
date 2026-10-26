// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
// 历史遗留实现，未加入 dram 目标；当前使用 MemoryObject，旧 Address 接口并不完整。
#include "dram/data_object.h"

#include <memory>
#include <vector>

#include "common/assert.h"
#include "common/type.h"

namespace llm_system {

/// @brief 旧地址对象构造接口，保存映射或地址描述；当前实现与头文件不一致且未加入构建。
DataObject::DataObject(MMap mmap) : mmap(mmap){};
/// @brief 旧地址对象构造接口，保存映射或地址描述；当前实现与头文件不一致且未加入构建。
DataObject::DataObject(const Address &addr) : addr_{addr} {};

/// @brief 旧接口：按块偏移除以每行列组数计算行偏移；使用已过时的 addr_ 字段。
const int DataObject::GetRowAddr(int offset) {
  assertTrue(offset < addr_.num_chunks_, "not valid offset");
  int row_stride = offset / addr_.num_column_groups_;
  return addr_.row_start_index_ + row_stride;
}

/// @brief 旧接口：按块偏移对每行列组数取模计算列组偏移；使用已过时的 addr_ 字段。
const int DataObject::GetColAddr(int offset) {
  assertTrue(offset < addr_.num_chunks_, "not valid offset");
  int column_stride = offset % addr_.num_column_groups_;
  return addr_.column_group_start_index_ + column_stride;
}

}  // namespace llm_system
