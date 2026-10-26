// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#pragma once

#include <tuple>
#include <vector>
#include "dram/dram_type.h"

namespace llm_system {

/// @brief 性能模型使用的处理器类别：GPU、LOGIC 或 PIM；不是宿主机真实执行线程的类型。
enum class ProcessorType { NONE = 0, PIM, LOGIC, GPU, MAX };

/// @brief 一次算子执行的附加描述：指定候选处理器、并行方式及注意力/批量矩阵操作所需参数。
struct LayerInfo {
  std::vector<ProcessorType> processor_type;
  int attention_group_size = 8;
  int num_heads = 64;
  int num_kv_heads = 8;
  int head_dim = 1;
  int qk_rope_head_dim = 0;
  int kv_lora_rank = 0;
  int parallel_num = 0;
  int decoder_idx = 0;
  bool parallel_execution = false;
  bool duplicated_input = false;
  bool use_flash_mla = true;
  bool use_flash_attention = true;
};

/// @brief 算子分派标识，Executor 据此选择线性层、激活或注意力等性能估算入口。
enum class LayerType {
  LINEAR = 0,
  BATCHED_LINEAR,
  ACTIVATION,
  ATTENTION_GEN,
  ATTENTION_SUM,
  ATTENTION_MIXED,
  MLA_GEN,
  MLA_SUM,
  MLA_MIXED,
  ABSORBED_MLA_GEN,
  ABSORBED_MLA_SUM,
  MAX
};

// forward declaration
/// @brief DRAMRequest 的前向声明：此处只声明类型，完整结构见对应头文件。
class DRAMRequest;
/// @brief Tensor 的前向声明：此处只声明类型，完整结构见对应头文件。
class Tensor;
/// @brief Device 的前向声明：此处只声明类型，完整结构见对应头文件。
class Device;
/// @brief TopModuleGraph 的前向声明：此处只声明类型，完整结构见对应头文件。
class TopModuleGraph;
/// @brief Cluster 的前向声明：此处只声明类型，完整结构见对应头文件。
class Cluster;
/// @brief Module 的前向声明：此处只声明类型，完整结构见对应头文件。
class Module;
/// @brief DRAMInterface 的前向声明：此处只声明类型，完整结构见对应头文件。
class DRAMInterface;

/// @brief MMapController 的前向声明：此处只声明类型，完整结构见对应头文件。
class MMapController;
/// @brief Device 的前向声明：此处只声明类型，完整结构见对应头文件。
class Device;

using Tensor_Ptr = std::shared_ptr<Tensor>;
using DRAMRequest_Ptr = std::shared_ptr<DRAMRequest>;
using Device_Ptr = std::shared_ptr<Device>;

using TopModuleGraph_ptr = std::shared_ptr<TopModuleGraph>;
using Module_ptr = std::shared_ptr<Module>;
using Cluster_ptr = std::shared_ptr<Cluster>;
using DRAMInterface_Ptr = std::shared_ptr<DRAMInterface>;

using MMapController_Ptr = std::shared_ptr<MMapController>;
using Device_Ptr = std::shared_ptr<Device>;

using CacheKey = std::tuple<LayerType, ProcessorType, DRAMRequestType, long>;

}  // namespace llm_system