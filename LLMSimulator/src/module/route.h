// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "module/base.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 专家负载更新模块：调用 batch 的专家采样/轨迹统计，并汇集参与设备的专家 token 数。
class GateUpdate : public Module {
  // Route tensor in to tensor vec
 public:
  using Ptr = std::shared_ptr<GateUpdate>;

  /// @brief GateUpdate 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new GateUpdate(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };
  /// @brief 从各参与设备的 DP batch 更新并汇总专家 token 数，供本设备路由和专家计算量估算。
  void aggregate_expert(BatchedSequence::Ptr sequences_metadata);
 private:
  /// @brief 初始化 GateUpdate 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  GateUpdate(std::string& prefix, std::string& name,
        std::vector<int> device_list, Device::Ptr device);
  /// @brief GateUpdate 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  GateUpdate() = default;

  /// @brief 更新跨设备专家负载统计，并返回与输入同形状的激活描述。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 专家路由描述模块：按每个专家的 token 数构造输入形状，并设置 GPU/低算力单元执行标记。
class Route : public Module {
  // Route tensor in to tensor vec
 public:
  using Ptr = std::shared_ptr<Route>;

  /// @brief Route 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int num_expert_per_device,
                                  int expert_offset,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new Route(prefix, name, num_expert_per_device, expert_offset, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Route 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Route(std::string& prefix, std::string& name, int num_expert_per_deice, int expert_offset, 
        std::vector<int> device_list, Device::Ptr device);
  /// @brief Route 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Route() = default;
  
  int expert_offset;
  int num_expert_per_device;

  /// @brief 按本设备专家的 token 数建立输入分片；结合并行/异构配置选择高低执行标记或候选性能比较。
  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system