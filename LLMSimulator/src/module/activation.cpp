// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/activation.h"

#include "common/assert.h"
namespace llm_system {

/// @brief 初始化 SiluAndMul 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
SiluAndMul::SiluAndMul(std::string& prefix, std::string& name,
                       Device::Ptr device)
    : Module(prefix, name, device, {}, true), activation_factor(1) {
  std::vector<int> shape = {1, 1};
  Tensor::Ptr output = Tensor::Create("SiluOut", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
};

/// @brief 检查特征维度能被 activation_factor 整除并返回对应输出形状；当前不提交性能估算。
Tensor::Ptr SiluAndMul::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  float time = 0;
  assertTrue(input->shape[1] % activation_factor == 0,
             "input.dim(1) % this->activation_factor != 0");
  std::vector<int> shape = {input->shape[0],
                            input->shape[1] / activation_factor};
  Tensor::Ptr output = get_activation("SiluOut", shape);
  return output;
}

/// @brief 初始化 Activation 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
Activation::Activation(std::string& prefix, std::string& name,
                       int activation_factor, Device::Ptr device)
    : Module(prefix, name, device, {}, true),
      activation_factor(activation_factor) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output = Tensor::Create("ActOut", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
  Tensor::Ptr gate_output = Tensor::Create("GateOut", shape, "act", device, device->model_config.precision_byte);
  add_tensor(gate_output);
};

/// @brief 按 activation_factor 更新输出，传播异构执行标记并提交 ACTIVATION 性能估算。
Tensor::Ptr Activation::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  float time = 0;
  assertTrue(input->shape[1] % activation_factor == 0,
             "input.dim(1) % this->activation_factor != 0");

  std::vector<int> shape = {input->shape[0],
                            input->shape[1] / activation_factor};
  Tensor::Ptr output = get_activation("ActOut", shape);
  Tensor::Ptr gate_output = get_activation("GateOut", shape);

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.push_back(gate_output);
  tensor_list.push_back(input);
  tensor_list.push_back(output);

  output->perform_with_optimal = input->perform_with_optimal;

  LayerInfo info;
  info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    info.parallel_execution = true;
    if (input->isPerformHigh()) {
      info.processor_type = {device->config.high_processor_type};
      output->setPerformHigh();
    } else {
      info.processor_type = {device->config.low_processor_type};
      output->setPerformLow();
    }
  } else if (device->config.hetero_subbatch) {
    info.processor_type = {device->config.high_processor_type};
  } else if (device->config.processor_type.size() != 1) {
    if (!input->perform_with_optimal) {
      info.processor_type = {device->config.high_processor_type};
    }
  }
  device->execution(LayerType::ACTIVATION, tensor_list, sequences_metadata,
                    info);

  return output;
}

}  // namespace llm_system
