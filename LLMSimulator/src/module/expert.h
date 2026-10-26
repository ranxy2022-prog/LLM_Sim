// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/parallel.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

/// @brief 专家前馈网络组合层：组织专家路由、分发、各专家计算和结果收集。
class ExpertFFN : public Module {
 public:
  using Ptr = std::shared_ptr<ExpertFFN>;
  /// @brief ExpertFFN 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new ExpertFFN(prefix, name, model_config, scheduler,
                                device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 ExpertFFN 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  ExpertFFN(std::string& prefix, std::string& name,
            const ModelConfig& model_config, Scheduler::Ptr scheduler,
            std::vector<int>& device_list, Device::Ptr device);
  /// @brief ExpertFFN 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  ExpertFFN() = default;

  /// @brief 组织专家 token 分发、路由、各专家前馈和结果收集，并处理配置的共享专家路径。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  int expert_offset;
  int num_expert_per_device;
  int num_shared_expert;
  std::string expert_file_path;
};

}  // namespace llm_system