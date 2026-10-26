// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。

#include "lm_head.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
// LmHead //

namespace llm_system {

/// @brief 初始化 LmHead 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
LmHead::LmHead(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;
  std::vector<int> wgt_shape = {hidden_dimension, n_vocab};
  std::vector<int> act_shape = {1, hidden_dimension};

  Tensor::Ptr LmHead = Tensor::Create("lm_head_wgt", wgt_shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr output = Tensor::Create("Hidden vector", act_shape, "act", device, device->model_config.precision_byte);
  add_tensor(LmHead);
  add_tensor(output);
}

/// @brief 用本轮总 token 数和词表大小更新输出形状；当前不执行词表矩阵乘或单独记录其耗时。
Tensor::Ptr LmHead::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = sequences_metadata->get_process_token();
  int n = model_config.n_vocab;

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Hidden vector", shape);

  return output;
}

}  // namespace llm_system