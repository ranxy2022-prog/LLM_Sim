// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <memory>
#include <vector>

#include "common/type.h"
#include "hardware/base.h"
#include "module/base.h"

namespace llm_system {

/// @brief Tensor 的前向声明：此处只声明类型，完整结构见对应头文件。
class Tensor;

/// @brief 一次算子/内存请求的性能结果：耗时为 ns，包含 FLOPs、字节数、利用率及 DRAM 命令计数。
struct ExecStatus {
  ProcessorType processor_type;
  time_ns time = 0.0;
  time_ns total_duration = 0.0;
  time_ns compute_duration = 0.0;
  time_ns memory_duration = 0.0;

  hw_metric flops = 0.0;
  hw_metric memory_size = 0.0;

  hw_metric opb = 0.0;
  util compute_util = 0.0;
  util memory_util = 0.0;
  // time_ns pim_duration = 0.0;

  counter_t generic_read_cmd = 0.0;
  counter_t generic_write_cmd = 0.0;
  counter_t compute_pim_cmd = 0.0;
  counter_t move_pim_cmd = 0.0;

  counter_t act_count = 0;
  counter_t read_count = 0;
  counter_t write_count = 0;
  counter_t all_act_count = 0;
  counter_t all_read_count = 0;
  counter_t all_write_count = 0;
  counter_t ref_count = 0;

  bool parallel_execution = false;

  /// @brief 累加内存耗时和部分命令计数；不会自动累加 total_duration/compute_duration 等所有字段，调用者需另行设置。
  ExecStatus& operator+=(const ExecStatus& rhs) {
    // 这是部分字段合并，不能把它当作整个 ExecStatus 的逐字段加法。
    // 注意原实现 move_pim_cmd 加的是 rhs.compute_pim_cmd；此处只解释原行为，不修改算法。
    memory_duration += rhs.memory_duration;
    generic_read_cmd += rhs.generic_read_cmd;
    compute_pim_cmd += rhs.compute_pim_cmd;
    move_pim_cmd += rhs.compute_pim_cmd;

    act_count += rhs.act_count;
    read_count += rhs.read_count;
    write_count += rhs.write_count;
    all_act_count += rhs.all_act_count;
    all_read_count += rhs.all_read_count;
    all_write_count += rhs.all_write_count;
    ref_count += rhs.ref_count;

    return *this;
  }
};

/// @brief 设备执行的累计状态：保存普通/高低处理器时间线、能耗和当前张量信息，供模块时间戳采样。
class StatusBoard {
 public:
  using Tensor_Ptr = std::shared_ptr<Tensor>;
  /// @brief StatusBoard 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  StatusBoard() = default;

  // accumulate
  time_ns device_time = 0;

  time_ns high_time = 0;  // time of GPU
  time_ns low_time = 0;   // time of PIM

  Tensor_Ptr tensor;

  TensorVec tensor_vec;
  // TensorVec output_tensor_vec;

  // energy_nJ act;
  // energy_nJ read;
  // energy_nJ write;
  // energy_nJ tsv;
  // energy_nJ interposer_io;
  energy_nJ act_energy;
  energy_nJ read_energy;
  energy_nJ write_energy;

  energy_nJ all_act_energy;
  energy_nJ all_read_energy;
  energy_nJ all_write_energy;

  energy_nJ act_energy_load;
  energy_nJ read_energy_load;
  energy_nJ write_energy_load;

  energy_nJ all_act_energy_load;
  energy_nJ all_read_energy_load;
  energy_nJ all_write_energy_load;

  energy_nJ mac_energy;

  util compute_util;
  util memory_util;

  // record
  time_ns start_time = 0;
  time_ns end_time = 0;

  hw_metric flops = 0.0;
  hw_metric memory_size = 0.0;
  hw_metric opb = 0.0;
  bool isTensorVec;
  std::vector<int> input_tensor_shape;
  std::vector<int> output_tensor_shape;
  std::vector<std::vector<int>> input_tensor_vec_shape;
  std::vector<std::vector<int>> output_tensor_vec_shape;
  ProcessorType processor_type;
  bool parallel_execution = false;
};

}  // namespace llm_system
