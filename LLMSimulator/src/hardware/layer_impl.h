// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#pragma once
#include <memory>
#include <tuple>

#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/base.h"
#include "module/status.h"
#include "model/model_config.h"
#include "scheduler/sequence.h"
namespace llm_system {

/// @brief 先查执行缓存，未命中时为张量创建逻辑内存请求并调用 DRAM 仿真，返回统计并缓存；缓存键不含完整地址历史。
ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, Tensor_Ptr tensor);

/// @brief 通过设备理想内存路径估算命令计数并返回统计，供未启用 Ramulator 的能耗估算使用。
ExecStatus getIdealMemoryStatus(Device_Ptr device, ProcessorType processor_type,
                          DRAMRequestType dram_request_type, Tensor_Ptr tensor);

/// @brief 估算线性层矩阵乘在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus LinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator);

/// @brief 估算线性层矩阵乘在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus LinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr ouptut,
                                bool use_ramulator);

/// @brief 估算线性层矩阵乘在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus LinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator);

/// @brief 估算批量矩阵乘在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus BatchedLinearExecutionGPU(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator, bool duplicated_input);

/// @brief 估算批量矩阵乘在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus BatchedLinearExecutionLogic(Device_Ptr device, Tensor_Ptr input,
                                Tensor_Ptr weight, Tensor_Ptr ouptut,
                                bool use_ramulator, bool duplicated_input);

/// @brief 估算批量矩阵乘在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus BatchedLinearExecutionPIM(Device_Ptr device, Tensor_Ptr input,
                              Tensor_Ptr weight, Tensor_Ptr ouptut,
                              bool use_ramulator, bool duplicated_input);

/// @brief 估算激活在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus ActivationExecutionGPU(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator);

/// @brief 估算激活在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus ActivationExecutionLogic(Device_Ptr device, Tensor_Ptr gate_output,
                                    Tensor_Ptr input, Tensor_Ptr output,
                                    bool use_ramulator);

/// @brief 估算激活在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus ActivationExecutionPIM(Device_Ptr device, Tensor_Ptr gate_output,
                                  Tensor_Ptr input, Tensor_Ptr output,
                                  bool use_ramulator);

/// @brief 估算常规 decode 注意力在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算常规 decode 注意力在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算常规 decode 注意力在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算常规 prefill 注意力在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算常规 prefill 注意力在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AttentionSumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算常规 prefill 注意力在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AttentionSumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 逐请求累计注意力 scoring 与 context 的 roofline 耗时并返回利用率；当前此混合阶段实现未使用 use_ramulator 参数，不产生详细内存请求轨迹。
ExecStatus AttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 常规混合阶段注意力的 Logic 占位实现：当前返回默认 ExecStatus，没有估算算子成本。
ExecStatus AttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor_list,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator);

/// @brief 逐请求累计注意力 scoring 与 context 的 roofline 耗时并返回利用率；当前此混合阶段实现未使用 use_ramulator 参数，不产生详细内存请求轨迹。
ExecStatus AttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算非吸收 MLA decode 注意力在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus MultiLatentAttentionGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算非吸收 MLA decode 注意力在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus MultiLatentAttentionGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算非吸收 MLA decode 注意力在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus MultiLatentAttentionGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算非吸收 MLA prefill 注意力在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus MultiLatentAttentionSumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算非吸收 MLA prefill 注意力在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus MultiLatentAttentionSumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算非吸收 MLA prefill 注意力在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus MultiLatentAttentionSumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 逐请求累计注意力 scoring 与 context 的 roofline 耗时并返回利用率；当前此混合阶段实现未使用 use_ramulator 参数，不产生详细内存请求轨迹。
ExecStatus MultiLatentAttentionMixedExecutionGPU(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 非吸收 MLA 混合阶段注意力的 Logic 占位实现：当前返回默认 ExecStatus，没有估算算子成本。
ExecStatus MultiLatentAttentionMixedExecutionLogic(Device_Ptr device,
                                        std::vector<Tensor_Ptr> tensor_list,
                                        BatchedSequence::Ptr sequences_metadata,
                                        LayerInfo layer_info,
                                        bool use_ramulator);

/// @brief 逐请求累计注意力 scoring 与 context 的 roofline 耗时并返回利用率；当前此混合阶段实现未使用 use_ramulator 参数，不产生详细内存请求轨迹。
ExecStatus MultiLatentAttentionMixedExecutionPIM(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算吸收 MLA decode 注意力在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AbsorbMLAGenExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算吸收 MLA decode 注意力在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AbsorbMLAGenExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算吸收 MLA decode 注意力在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AbsorbMLAGenExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算吸收 MLA prefill 注意力在 GPU 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AbsorbMLASumExecutionGPU(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

/// @brief 估算吸收 MLA prefill 注意力在 Logic 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AbsorbMLASumExecutionLogic(Device_Ptr device,
                                      std::vector<Tensor_Ptr> tensor_list,
                                      BatchedSequence::Ptr sequences_metadata,
                                      LayerInfo layer_info, bool use_ramulator);

/// @brief 估算吸收 MLA prefill 注意力在 PIM 上的成本：按形状/精度统计运算和访存，结合对应算力、带宽及可选内存仿真返回 ExecStatus；时间单位 ns。
ExecStatus AbsorbMLASumExecutionPIM(Device_Ptr device,
                                    std::vector<Tensor_Ptr> tensor_list,
                                    BatchedSequence::Ptr sequences_metadata,
                                    LayerInfo layer_info, bool use_ramulator);

}  // namespace llm_system