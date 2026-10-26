// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/layernorm.h"
#include "module/residual.h"
#include "module/parallel.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

/// @brief 普通 Transformer decoder 组合层：组织注意力、归一化、残差与稠密前馈网络。
class Decoder : public Module {
 public:
  using Ptr = std::shared_ptr<Decoder>;
  /// @brief Decoder 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Decoder(prefix, name, model_config, scheduler,
                              device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Decoder 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Decoder(std::string& prefix, std::string& name,
          const ModelConfig& model_config, Scheduler::Ptr scheduler,
          std::vector<int>& device_list, Device::Ptr device);
  /// @brief Decoder 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Decoder() = default;

  /// @brief 依序调用普通 decoder 的注意力、归一化、残差与前馈子模块，构成一层执行路径。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief MoE decoder 组合层：组织注意力、归一化、残差和专家前馈网络。
class MoEDecoder : public Module {
 public:
  using Ptr = std::shared_ptr<MoEDecoder>;
  /// @brief MoEDecoder 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEDecoder(prefix, name, model_config, scheduler,
                                 device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 MoEDecoder 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  MoEDecoder(std::string& prefix, std::string& name,
             const ModelConfig& model_config, Scheduler::Ptr scheduler,
             std::vector<int>& device_list, Device::Ptr device);
  /// @brief MoEDecoder 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  MoEDecoder() = default;

  /// @brief 依序调用 MoE decoder 的注意力、归一化、残差与专家网络子模块。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system