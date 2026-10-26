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

/// @brief 词表输出头描述模块：登记输出权重并产生词表维度形状；当前 forward 不实际计算 logits 或单独计时。
class LmHead : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<LmHead>;

  /// @brief LmHead 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  ModelConfig model_config,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new LmHead(prefix, name, model_config, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 LmHead 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  LmHead(std::string& prefix, std::string& name, ModelConfig model_config,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief LmHead 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  LmHead() = default;

  /// @brief 用本轮总 token 数和词表大小更新输出形状；当前不执行词表矩阵乘或单独记录其耗时。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

  ModelConfig model_config;
};

}  // namespace llm_system