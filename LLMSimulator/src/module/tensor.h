// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <cassert>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#include "common/assert.h"
#include "dram/memory_object.h"
#include "hardware/device.h"
#include "scheduler/sequence.h"

/// @brief Module 的前向声明：此处只声明类型，完整结构见对应头文件。
class Module;
namespace llm_system {

/// @brief 张量元数据对象：保存形状、字节精度、weight/act/cache 标签、就绪标志和内存映射；不保存实际数值数组。
class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  using Ptr = std::shared_ptr<Tensor>;
  using Module_ptr = std::shared_ptr<Module>;

  /// @brief 创建张量元数据，再根据所属设备建立 MemoryObject；需要设备/映射器可用，不创建数值数组。
  [[nodiscard]] static Ptr Create(std::string name = "dummy",
                                  std::vector<int> shape = {1, 1},
                                  std::string tag = "act",
                                  Device::Ptr device = NULL,
                                  int precision_byte = 1,
                                  MMap mmap = MMap::ALL_CHANNEL) {
    Ptr ptr = Ptr(new Tensor(name, shape, tag, device, precision_byte, mmap));
    ptr->setMemoryObject();
    return ptr;
  }

  std::string name;
  std::vector<int> shape;
  std::string tag;
  int precision_byte;
  bool ready;

  // for MoE layers
  bool perform_with_optimal;
  
  // for parallel execution
  bool parallel_execution;
  bool perform_at_high;

  bool isDecoder = false;
  bool isMemPoolLoad = false;

  /// @brief 返回指定维度的长度，用于模块检查和形状推导。
  int dim(int i);

  /// @brief 将张量标记为就绪，并按当前形状更新 MemoryObject 的逻辑大小，允许消费者继续执行。
  void set();
  /// @brief 清除张量就绪标志，为下一轮图执行重新建立生产者/消费者顺序。
  void unset();

  /// @brief 将各维长度相乘再乘 precision_byte，返回张量的逻辑字节数。
  long getSize();

  /// @brief 返回形状向量的可修改引用；若要同步 MemoryObject 大小，应使用 setShape。
  std::vector<int>& getShape() { return shape; }

  /// @brief 更新张量所属设备指针。
  void set_device(Device::Ptr device);
  /// @brief 返回张量关联的模拟设备。
  Device::Ptr get_device() { return device; }

  /// @brief 把张量名称、维度个数和标签格式化为字符串；此函数不输出各维长度，日志形状由时间戳打印函数处理。
  std::string ToString();
  /// @brief 取得所属模块的完整名称，供跨设备依赖匹配。
  std::string get_module_map_name();

  /// @brief 把张量标记为参与异构并行并由高算力单元处理。
  void setPerformHigh() {
    parallel_execution = true;
    perform_at_high = true;
  }
  /// @brief 把张量标记为参与异构并行并由低算力单元处理。
  void setPerformLow() {
    parallel_execution = true;
    perform_at_high = false;
  }

  /// @brief 读取高/低算力单元执行标志，供分派和时间线累计使用。
  bool isPerformHigh() { return perform_at_high; }

  /// @brief 通过 shared_from_this 返回当前 Tensor 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Ptr getPtr() { return shared_from_this(); }

  /// @brief 返回张量的模拟内存分布方式。
  MMap getMMap() { return mmap; }
  /// @brief 返回张量关联的模拟地址区间描述。
  MemoryObject::Ptr getMemoryObject() { return memory_object; }
  /// @brief 记录张量所属模块，供完整名称与依赖查询。
  void set_module(Module_ptr _module) { module = _module; }

  bool timeboard_synced;

  /// @brief 请求所属设备为该张量建立模拟内存映射。
  void setMemoryObject();

  // set shape and update memory object size
  /// @brief 更新形状，并同步修改 MemoryObject 的逻辑大小和访问块数。
  void setShape(std::vector<int> shape_) {
    shape = shape_;
    memory_object->setSize(getSize());
  }

  /// @brief 关联已有的内存地址对象；此处仅保存指针，不执行真实内存分配。
  void allocateMemoryObject(MemoryObject::Ptr memory_object_) {
    memory_object = memory_object_;
  };

 private:
  Device::Ptr device;
  Module_ptr module;
  MMap mmap;
  MemoryObject::Ptr memory_object;
  /// @brief 保存名称、形状、精度、标签和设备，初始化就绪及异构执行标记。
  Tensor(std::string name, std::vector<int> shape, std::string tag,
         Device::Ptr device, int precision_byte, MMap mmap);
};

}  // namespace llm_system