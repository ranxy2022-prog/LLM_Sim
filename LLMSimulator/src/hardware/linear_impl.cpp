// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#include <memory>

#include "common/type.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "hardware/layer_impl.h"
#include "module/tensor.h"

namespace llm_system {
/// @brief DRAMRequest 的前向声明：此处只声明类型，完整结构见对应头文件。
class DRAMRequest;
/// @brief Tensor 的前向声明：此处只声明类型，完整结构见对应头文件。
class Tensor;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;

/// @brief 估算线性层矩阵乘在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  // 一次乘加记作 2 FLOPs；以下访问量假设输入/权重各读一次、输出写一次。
  // 这不是 tile 级访存轨迹，没有表达缓存命中、重复装载或真实权重布局导致的 row conflict。
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;
  
  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);
  // roofline 近似：用计算与访存两者较慢的一方作为总时间，而非把两项直接相加。

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  return exec_status;
};

/// @brief 估算线性层矩阵乘在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

/// @brief 估算线性层矩阵乘在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

/// @brief 估算批量矩阵乘在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  assertTrue(input->shape.size() == 3, "input tensor is not 3D tensor");
  assertTrue(weight->shape.size() == 3, "weight tensor is not 3D tensor");
  assertTrue(output->shape.size() == 3, "output tensor is not 3D tensor");

  auto config = device->config;
  hw_metric compute_peak_flops = config.compute_peak_flops;
  hw_metric memory_bandwidth = config.memory_bandwidth;

  int num_heads = input->shape[0];
  std::vector<int> input_orig_shape = input->shape;
  std::vector<int> weight_orig_shape = weight->shape;
  std::vector<int> output_orig_shape = output->shape;

  int m = input->shape[1];
  int k = input->shape[2];
  int n = weight->shape[2];

  hw_metric total_flops = 2.0 * m * k * n * 1.0 * num_heads;
  
  hw_metric total_memory_size;
  if(duplicated_input){
    total_memory_size = 1.0 * (m * k + k * n * 1.0  * num_heads + m * n * 1.0  * num_heads) * weight->precision_byte;
  }
  else{
    total_memory_size = 1.0 * (m * k + k * n + m * n) * 1.0  * num_heads * weight->precision_byte;
  }

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }

  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    if(duplicated_input){
      input->setShape({m, k});
    }
    else{
      input->setShape({m, k * num_heads});
    }
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    
    weight->setShape({k * num_heads, n});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, weight);

    output->setShape({m, n * num_heads});
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  exec_status.opb = total_flops / total_memory_size;

  input->setShape(input_orig_shape);
  weight->setShape(weight_orig_shape);
  output->setShape(output_orig_shape);

  return exec_status;
};

/// @brief 估算批量矩阵乘在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr output,
                                bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops =
      config.logic_memory_bandwidth * config.logic_op_b;
  hw_metric memory_bandwidth = config.logic_memory_bandwidth;
  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::LOGIC,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::LOGIC, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

/// @brief 估算批量矩阵乘在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr output,
                              bool use_ramulator, bool duplicated_input) {
  auto config = device->config;
  hw_metric compute_peak_flops = config.pim_memory_bandwidth * config.pim_op_b;
  hw_metric memory_bandwidth = config.pim_memory_bandwidth;

  if(input->precision_byte == 1){
    compute_peak_flops *= 2;
  }

  double m = input->shape[0];
  double k = input->shape[1];
  double n = weight->shape[1];

  hw_metric total_flops = 2.0 * m * k * n;
  hw_metric total_memory_size = (m * k + k * n + m * n) * weight->precision_byte;

  time_ns compute_duration =
      total_flops / compute_peak_flops * 1000 * 1000 * 1000;
  time_ns memory_duration =
      total_memory_size / memory_bandwidth * 1000 * 1000 * 1000;

  ExecStatus exec_status;
  if (input->getSize() == 0) {
    return exec_status;
  }
  exec_status.compute_duration = compute_duration;

  if (use_ramulator) {
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kRead, PIMOperandType::kDRAM, input);
    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::PIM,
                       DRAMRequestType::kGEMV, PIMOperandType::kSrc, weight);

    exec_status +=
        issueRamulator(device, LayerType::LINEAR, ProcessorType::GPU,
                       DRAMRequestType::kWrite, PIMOperandType::kDRAM, output);
  } else {
    exec_status.memory_duration = memory_duration;
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kRead, input);

    exec_status += getIdealMemoryStatus(device, ProcessorType::PIM, DRAMRequestType::kRead, weight);
    
    exec_status += getIdealMemoryStatus(device, ProcessorType::GPU, DRAMRequestType::kWrite, output);
  }

  exec_status.total_duration =
      std::max(exec_status.compute_duration, exec_status.memory_duration);

  exec_status.compute_util = 1000.0 * 1000.0 * 1000.0 * total_flops /
                             compute_peak_flops / exec_status.total_duration;
  exec_status.memory_util = 1000.0 * 1000.0 * 1000.0 * total_memory_size /
                            memory_bandwidth / exec_status.total_duration;

  exec_status.flops = total_flops;
  exec_status.memory_size = total_memory_size;

  return exec_status;
};

}  // namespace llm_system
