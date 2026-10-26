// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/residual.h"
#include "scheduler/scheduler.h"

#include "common/assert.h"
// AllReduce //

namespace llm_system {

/// @brief 初始化 Residual 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
Residual::Residual(std::string& prefix, std::string& name,
                     int hidden_dim, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) { // sync == true, thererfore need to be synced
    std::vector<int> shape = {1, 1};

    Tensor::Ptr residual_out = Tensor::Create("residual_out", shape, "act", device, device->model_config.precision_byte);
    add_tensor(residual_out);
}

/// @brief 按 M*K 加法、三份张量访存量估算耗时，更新设备及必要的高/低时间线。
Tensor::Ptr Residual::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];

  Tensor::Ptr output = get_activation("residual_out", input->shape);

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  hw_metric compute_peak_flops = device->config.compute_peak_flops;
  hw_metric memory_bandwidth = device->config.memory_bandwidth;

  double flops, memory_size;
  flops = m * k * 1;
  memory_size = (3.0 * m * k) * input->precision_byte; // input x 2, store x 1

  time_ns compute_duration = flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration = memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = std::max(compute_duration, memory_duration);

  if (input->parallel_execution) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

}  // namespace llm_system