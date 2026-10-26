// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iostream>
#include <string>

#include "model/model_config.h"
#include "module/activation.h"
#include "module/parallel.h"
#include "module/compressed_kv_restore.h"
#include "module/layernorm.h"
#include "module/rope.h"
#include "module/tensor.h"
#include "scheduler/scheduler.h"

namespace llm_system {

/// @brief 常规注意力组合层：组织投影、位置编码、注意力计算及输出投影等子模块。
class Attention : public Module {
 public:
  using Ptr = std::shared_ptr<Attention>;
  /// @brief Attention 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Attention(prefix, name, model_config, scheduler,
                                device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Attention 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Attention(std::string& prefix, std::string& name,
            const ModelConfig& model_config, Scheduler::Ptr scheduler,
            std::vector<int>& device_list, Device::Ptr device);
  /// @brief 初始化 Attention 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Attention(){};

  /// @brief 依序调用常规注意力子模块，传播投影、位置编码、注意力与输出投影的形状和性能描述。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief MLA 组合层：按 use_absorb 等配置组织低秩投影、KV 恢复或权重吸收路径。
class MultiLatentAttention : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttention>;
   /// @brief MultiLatentAttention 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   const ModelConfig& model_config,
                                   Scheduler::Ptr scheduler,
                                   std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttention(prefix, name, model_config, scheduler,
                                 device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
   /// @brief 初始化 MultiLatentAttention 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
   MultiLatentAttention(std::string& prefix, std::string& name,
             const ModelConfig& model_config, Scheduler::Ptr scheduler,
             std::vector<int>& device_list, Device::Ptr device);
   /// @brief 初始化 MultiLatentAttention 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
   MultiLatentAttention(){};
 
   /// @brief 按 use_absorb 选择 MLA 数据路径，组织低秩投影、位置编码、注意力及输出投影调用。
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  
   bool use_absorb;
   int num_heads;
   int parallel_num;
 };

/// @brief 两段线性投影构成的前馈网络组合层，组织激活与可选的结果归约。
class FeedForward2Way : public Module {
 public:
  using Ptr = std::shared_ptr<FeedForward2Way>;
  /// @brief FeedForward2Way 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device,
                                  bool perform_all_reduce = true,
                                  bool is_expert = false,
                                  bool use_dp = false) {
    Ptr ptr = Ptr(new FeedForward2Way(prefix, name, model_config, scheduler,
                                      device_list, device, perform_all_reduce,
                                      is_expert, use_dp));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 FeedForward2Way 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  FeedForward2Way(std::string& prefix, std::string& name,
                  const ModelConfig& model_config, Scheduler::Ptr scheduler,
                  std::vector<int>& device_list, Device::Ptr device,
                  bool perform_all_reduce, bool is_expert, bool use_dp);
  /// @brief FeedForward2Way 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  FeedForward2Way() = default;

  /// @brief 依序调用前馈网络的线性投影与激活，按配置执行输出归约。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  bool perform_all_reduce;
};

/// @brief 门控前馈网络组合层：组织 gate/up/down 投影、激活与可选归约。
class FeedForward3Way : public Module {
 public:
  using Ptr = std::shared_ptr<FeedForward3Way>;
  /// @brief FeedForward3Way 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  const ModelConfig& model_config,
                                  Scheduler::Ptr scheduler,
                                  std::vector<int> device_list,
                                  Device::Ptr device,
                                  bool perform_all_reduce = true,
                                  bool is_expert = false,
                                  bool use_dp = false) {
    Ptr ptr = Ptr(new FeedForward3Way(prefix, name, model_config, scheduler,
                                      device_list, device, perform_all_reduce,
                                      is_expert, use_dp));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 FeedForward3Way 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  FeedForward3Way(std::string& prefix, std::string& name,
                  const ModelConfig& model_config, Scheduler::Ptr scheduler,
                  std::vector<int>& device_list, Device::Ptr device,
                  bool perform_all_reduce, bool is_expert, bool use_dp);
  /// @brief FeedForward3Way 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  FeedForward3Way() = default;

  /// @brief 依序调用门控前馈网络的投影与激活，按配置执行输出归约。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  bool perform_all_reduce;
};

}  // namespace llm_system