// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。

#include "embedding.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
// Embedding //

namespace llm_system {

/// @brief 初始化 Embedding 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
Embedding::Embedding(std::string& prefix, std::string& name,
                     ModelConfig model_config, std::vector<int> device_list,
                     Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      model_config(model_config) {
  int hidden_dimension = model_config.hidden_dim;
  int n_vocab = model_config.n_vocab;
  std::vector<int> wgt_shape = {n_vocab, hidden_dimension};
  std::vector<int> act_shape = {1, hidden_dimension};

  Tensor::Ptr embedding = Tensor::Create("Embedding", wgt_shape, "weight", device, device->model_config.precision_byte);
  Tensor::Ptr output = Tensor::Create("Hidden vector", act_shape, "act", device, device->model_config.precision_byte);
  add_tensor(embedding);
  add_tensor(output);
}

/// @brief 用本轮总 token 数和 hidden_dim 更新输出形状；不读取真实 token/权重值，也不单独估算 embedding 时间。
Tensor::Ptr Embedding::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = sequences_metadata->get_process_token();
  int n = model_config.hidden_dim;

  std::vector<int> shape = {m, n};
  Tensor::Ptr output = get_activation("Hidden vector", shape);

  return output;
}

}  // namespace llm_system