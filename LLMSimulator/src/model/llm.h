// 中文导读：模型结构与设备映射：组合算子模块、配置维度和并行组，在初始化阶段建立执行图。
#pragma once
#include "hardware/cluster.h"
#include "model/model_config.h"
#include "model/util.h"
#include "scheduler/scheduler.h"

namespace llm_system {

/// @brief 设备上的顶层语言模型模块：组织 embedding、各 decoder 层和输出头，并按模型配置选择结构。
class LLM : public Module {
 public:
  using Ptr = std::shared_ptr<LLM>;
  /// @brief LLM 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(const ModelConfig& model_config,
                                  Cluster::Ptr cluster,
                                  Scheduler::Ptr scheduler,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new LLM(model_config, cluster, scheduler, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 LLM 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  LLM(const ModelConfig& model_config, Cluster::Ptr cluster,
      Scheduler::Ptr scheduler, Device::Ptr device);

  /// @brief 按 embedding、decoder 和输出头顺序调用子模块；当前 decoder 循环未把每层 out 赋回 temp，勿当作数值推理实现。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

}  // namespace llm_system
