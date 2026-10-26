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

/// @brief 词嵌入描述模块：登记词表权重，根据本轮 token 数产生隐藏向量形状；当前 forward 不单独计时。
class Embedding : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<Embedding>;

  /// @brief Embedding 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  ModelConfig model_config,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new Embedding(prefix, name, model_config, device_list, device));
    ptr->set_tensor_module();
    return ptr;
    // return Ptr(new Embedding(name, input_size, output_size,
    //                                     device_list, device));
  };

 private:
  /// @brief 初始化 Embedding 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Embedding(std::string& prefix, std::string& name, ModelConfig model_config,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief Embedding 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Embedding() = default;

  /// @brief 用本轮总 token 数和 hidden_dim 更新输出形状；不读取真实 token/权重值，也不单独估算 embedding 时间。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

}  // namespace llm_system