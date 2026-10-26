// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 线性层模块：维护权重和输出形状，把矩阵乘法描述交给设备的算子性能模型。
class Linear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<Linear>;

  /// @brief Linear 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new Linear(prefix, name, input_size, output_size, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Linear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Linear(std::string& prefix, std::string& name, int input_size,
         int output_size, std::vector<int> device_list, Device::Ptr device);
  /// @brief Linear 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Linear() = default;

  /// @brief 检查 X[M,K] 与权重[K,N]，构造输出[M,N]，选择处理器标记并提交 LINEAR 性能估算。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 批量矩阵乘模块：组织多组输入/权重，支持输入复用与普通线性层实现分支。
class BatchedLinear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<BatchedLinear>;

  /// @brief BatchedLinear 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int num_batched_gemm, int input_size, int output_size,
                                  bool duplicated_input, bool use_plain_lienar, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(
        new BatchedLinear(prefix, name, num_batched_gemm, input_size, output_size, duplicated_input, use_plain_lienar, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 BatchedLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  BatchedLinear(std::string& prefix, std::string& name, int num_batched_gemm, int input_size,
         int output_size, bool duplicated_input, bool use_plain_lienar, std::vector<int> device_list, Device::Ptr device);
  /// @brief BatchedLinear 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  BatchedLinear() = default;

  /// @brief 组织输入/权重/输出张量列表，按实现分支调用普通或批量线性层性能模型。
  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;

  bool duplicated_input;
  bool use_plain_linear;
};

}  // namespace llm_system