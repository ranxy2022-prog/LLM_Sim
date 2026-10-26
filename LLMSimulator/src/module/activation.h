// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <cassert>
#include <iostream>
#include <string>

#include "module/module.h"
#include "module/tensor.h"

namespace llm_system {

/// @brief SiLU/门控乘法的形状占位模块；当前 forward 只构造输出描述，没有独立的硬件执行调用。
class SiluAndMul : public Module {
  // models suc has Mixtral, Llama, and Qwen use silu activation
 public:
  using Ptr = std::shared_ptr<SiluAndMul>;
  /// @brief SiluAndMul 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  Device::Ptr device) {
    return Ptr(new SiluAndMul(prefix, name, device));
  };

 private:
  /// @brief 初始化 SiluAndMul 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  SiluAndMul(std::string& prefix, std::string& name, Device::Ptr device);
  /// @brief 初始化 SiluAndMul 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  SiluAndMul(){};

  /// @brief 检查特征维度能被 activation_factor 整除并返回对应输出形状；当前不提交性能估算。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  int activation_factor;
};

/// @brief 激活模块：调整激活形状并向设备提交激活性能估算请求。
class Activation : public Module {
 public:
  using Ptr = std::shared_ptr<Activation>;
  /// @brief Activation 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int activation_factor, Device::Ptr device) {
    return Ptr(new Activation(prefix, name, activation_factor, device));
  };

 private:
  /// @brief 初始化 Activation 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Activation(std::string& prefix, std::string& name, int activation_factor,
             Device::Ptr device);
  /// @brief 初始化 Activation 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Activation(){};

  /// @brief 按 activation_factor 更新输出，传播异构执行标记并提交 ACTIVATION 性能估算。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  int activation_factor;
};

}  // namespace llm_system