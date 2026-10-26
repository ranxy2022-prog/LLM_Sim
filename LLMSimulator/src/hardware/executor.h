// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#pragma once
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_type.h"
#include "hardware/base.h"
#include "hardware/layer_impl.h"
#include "module/status.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 算子性能模型分派器：按算子与处理器类型调用估算函数，比较候选执行方案的耗时。
class Executor {
 public:
  std::vector<std::function<ExecStatus(Device_Ptr, Tensor_Ptr, Tensor_Ptr,
                                       Tensor_Ptr, bool)>>
      linear_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, Tensor_Ptr, Tensor_Ptr,
                                       Tensor_Ptr, bool, bool)>>
      batched_linear_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, Tensor_Ptr, Tensor_Ptr,
                                       Tensor_Ptr, bool)>>
      activation_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      attention_gen_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      attention_sum_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      attention_mixed_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      multi_latent_attention_gen_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      multi_latent_attention_sum_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      multi_latent_attention_mixed_function_ramulator_ptr;

  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      absorb_mla_gen_function_ramulator_ptr;
  std::vector<std::function<ExecStatus(Device_Ptr, std::vector<Tensor_Ptr>,
                                       BatchedSequence::Ptr, LayerInfo, bool)>>
      absorb_mla_sum_function_ramulator_ptr;

  /// @brief 针对算子配置的候选处理器计算执行成本，保留估计总耗时较小的结果并写回 Device。
  void execution(LayerType layer_type,
                 const std::vector<Tensor_Ptr>& tensor_list,
                 const BatchedSequence::Ptr sequences_metadata,
                 const std::vector<ProcessorType>& processor_type,
                 const LayerInfo layer_info, bool use_ramualtor,
                 Device_Ptr device);

  /// @brief 初始化各算子的 GPU/LOGIC/PIM 性能函数表，供后续按类型分派；此时不执行算子估算。
  Executor();

 private:
  /// @brief 初始化算子到处理器实现的函数表，建立 GPU/LOGIC/PIM 分派关系。
  void init();
  /// @brief 注册线性层矩阵乘在 GPU、LOGIC、PIM 上的性能估算函数。
  void initLinear();
  /// @brief 注册激活在 GPU、LOGIC、PIM 上的性能估算函数。
  void initActivation();
  /// @brief 注册常规 decode 注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initAttentionGen();
  /// @brief 注册常规 prefill 注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initAttentionSum();
  /// @brief 注册常规混合阶段注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initAttentionMixed();
  /// @brief 注册非吸收 MLA decode 注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initMultiLatentAttentionGen();
  /// @brief 注册非吸收 MLA prefill 注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initMultiLatentAttentionSum();
  /// @brief 注册非吸收 MLA 混合阶段注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initMultiLatentAttentionMixed();
  /// @brief 注册吸收 MLA decode 注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initAbsorbMLAGen();
  /// @brief 注册吸收 MLA prefill 注意力在 GPU、LOGIC、PIM 上的性能估算函数。
  void initAbsorbMLASum();

  /// @brief 设置 DRAM 侧处理器类型/带宽倍率，再按 LayerType 调用对应算子的执行分派函数。
  ExecStatus executePType(LayerType layer_type,
                          const std::vector<Tensor_Ptr>& tensor_list,
                          const BatchedSequence::Ptr sequences_metadata,
                          ProcessorType processor_type,
                          const LayerInfo layer_info, bool use_ramulator,
                          Device_Ptr device);

  /// @brief 从张量列表和 batch 元数据取得线性层矩阵乘参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeLinear(const std::vector<Tensor_Ptr>& tensor_list,
                           ProcessorType processor_type, bool use_ramulator,
                           Device_Ptr device);

  /// @brief 从张量列表和 batch 元数据取得批量矩阵乘参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeBatchedLinear(const std::vector<Tensor_Ptr>& tensor_list,
                           ProcessorType processor_type, bool use_ramulator,
                           bool duplicated_input, Device_Ptr device);

  /// @brief 从张量列表和 batch 元数据取得激活参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeActivation(const std::vector<Tensor_Ptr>& tensor_list,
                               ProcessorType processor_type, bool use_ramulator,
                               Device_Ptr device);

  /// @brief 从张量列表和 batch 元数据取得常规 decode 注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeAttentionGen(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  /// @brief 从张量列表和 batch 元数据取得常规 prefill 注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeAttentionSum(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  /// @brief 从张量列表和 batch 元数据取得常规混合阶段注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeAttentionMixed(
      const std::vector<Tensor_Ptr>& tensor_list,
      const BatchedSequence::Ptr sequences_metadata,
      ProcessorType processor_type, const LayerInfo layer_info,
      bool use_ramulator, Device_Ptr device);
  
  /// @brief 从张量列表和 batch 元数据取得非吸收 MLA decode 注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeMultiLatentAttentionGen(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  /// @brief 从张量列表和 batch 元数据取得非吸收 MLA prefill 注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeMultiLatentAttentionSum(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  /// @brief 从张量列表和 batch 元数据取得非吸收 MLA 混合阶段注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeMultiLatentAttentionMixed(
      const std::vector<Tensor_Ptr>& tensor_list,
      const BatchedSequence::Ptr sequences_metadata,
      ProcessorType processor_type, const LayerInfo layer_info,
      bool use_ramulator, Device_Ptr device);
  
  /// @brief 从张量列表和 batch 元数据取得吸收 MLA decode 注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeAbsorbMLAGen(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
  /// @brief 从张量列表和 batch 元数据取得吸收 MLA prefill 注意力参数，调用所选处理器的估算函数并返回 ExecStatus。
  ExecStatus executeAbsorbMLASum(const std::vector<Tensor_Ptr>& tensor_list,
                                 const BatchedSequence::Ptr sequences_metadata,
                                 ProcessorType processor_type,
                                 const LayerInfo layer_info, bool use_ramulator,
                                 Device_Ptr device);
                                 
};

};  // namespace llm_system
