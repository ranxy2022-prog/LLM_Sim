// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/tensor.h"

#include "common/assert.h"
#include "module/module.h"

namespace llm_system {
/// @brief 保存名称、形状、精度、标签和设备，初始化就绪及异构执行标记。
Tensor::Tensor(std::string name, std::vector<int> shape, std::string tag,
               Device::Ptr device, int precision_byte, MMap mmap)
    : name(name),
      shape(shape),
      tag(tag),
      device(device),
      precision_byte(precision_byte),
      mmap(mmap) {
  ready = false;
  timeboard_synced = false;

  perform_with_optimal = false;
  parallel_execution = false;
  perform_at_high = true;
}

/// @brief 将各维长度相乘再乘 precision_byte，返回张量的逻辑字节数。
long Tensor::getSize() {
  long size = precision_byte;
  for (int dim : shape) {
    size *= dim;
  }
  return size;
}

/// @brief 返回指定维度的长度，用于模块检查和形状推导。
int Tensor::dim(int i) {
  assert(i < shape.size());
  return shape[i];
}

/// @brief 将张量标记为就绪，并按当前形状更新 MemoryObject 的逻辑大小，允许消费者继续执行。
void Tensor::set() {
  ready = true;
  memory_object->setSize(getSize());
}

/// @brief 清除张量就绪标志，为下一轮图执行重新建立生产者/消费者顺序。
void Tensor::unset() { ready = false; }

/// @brief 请求所属设备为该张量建立模拟内存映射。
void Tensor::setMemoryObject() {
  assertFalse(device == nullptr, "Tensor is not allocated in device");
  device->setMemoryObject(getPtr());
}

/// @brief 更新张量所属设备指针。
void Tensor::set_device(Device::Ptr _device) { device = _device; }

/// @brief 取得所属模块的完整名称，供跨设备依赖匹配。
std::string Tensor::get_module_map_name() { return module->module_map_name; }

/// @brief 把张量名称、维度个数和标签格式化为字符串；此函数不输出各维长度，日志形状由时间戳打印函数处理。
std::string Tensor::ToString() {
  return "(" + name + ": " + std::to_string(shape.size()) + ", " + tag + ")";
}

}  // namespace llm_system