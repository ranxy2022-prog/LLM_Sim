// 中文导读：模型结构与设备映射：组合算子模块、配置维度和并行组，在初始化阶段建立执行图。
#include "model/llm.h"

#include "module/decoder.h"
#include "module/embedding.h"
#include "module/lm_head.h"
#include "module/layer.h"

namespace llm_system {

/// @brief 初始化 LLM 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
LLM::LLM(const ModelConfig& model_config, Cluster::Ptr cluster,
         Scheduler::Ptr scheduler, Device::Ptr device)
    : Module("", "LLM", device), model_config(model_config) {
  int ne_tp_dg = model_config.ne_tp_dg;
  std::vector<int> device_list;

  set_device_list(device_list, 0, cluster->num_total_device);

  auto embedding_layer = Embedding::Create(module_map_name, "Embedding_layer",
                                           model_config, device_list, device);
  add_module(embedding_layer);

  if ((model_config.model_name != "deepseekV3")) {
    for (int layer = 0; layer < model_config.num_layers; layer++) {
      if ((model_config.expert_freq != 0) && (layer % model_config.expert_freq == 0)) {
        // MoE decoder;
        auto moe_decoder = MoEDecoder::Create(
            module_map_name, "MoE_decoder_" + std::to_string(layer),
            model_config, scheduler, device_list, device);
        add_module(moe_decoder);

      } else {
        // non MoE decoder;
        // int ne_tp_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;
        // set_device_list(device_list, ne_tp_offset, ne_tp_dg);

        auto decoder =
            Decoder::Create(module_map_name, "decoder_" + std::to_string(layer),
                            model_config, scheduler, device_list, device);
        add_module(decoder);
      }
    }
  } else if ((model_config.model_name == "deepseekV3")) {
    for (int layer = 0; layer < model_config.first_k_dense; layer++) {
      auto decoder =
          Decoder::Create(module_map_name, "decoder_" + std::to_string(layer),
                          model_config, scheduler, device_list, device);
      add_module(decoder);
    }
    for (int layer = model_config.first_k_dense; layer < model_config.num_layers; layer++) {
      auto moe_decoder = MoEDecoder::Create(
          module_map_name, "MoE_decoder_" + std::to_string(layer), model_config,
          scheduler, device_list, device);
      add_module(moe_decoder);
    }
  } else {
    fail("No configuration of " + model_config.model_name);
  
  }
  auto lm_head = LmHead::Create(module_map_name, "lm_head",
    model_config, device_list, device);
  add_module(lm_head);
}

/// @brief 按 embedding、decoder 和输出头顺序调用子模块；当前 decoder 循环未把每层 out 赋回 temp，勿当作数值推理实现。
Tensor::Ptr LLM::forward(const Tensor::Ptr input,
                         BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr decoder;
  Tensor::Ptr temp, out;
  Module::Ptr next_decoder_loader;

  Module::Ptr embedding = get_module("Embedding_layer");
  temp = (*embedding)(input, sequences_metadata);

  if ((model_config.model_name != "deepseekV3")){
    for (int layer = 0; layer < model_config.num_layers; layer++) {
      if ((model_config.expert_freq != 0) && (layer % model_config.expert_freq == 0)){
        decoder = get_module("MoE_decoder_" + std::to_string(layer));
      } else {
        decoder = get_module("decoder_" + std::to_string(layer));
      }
      out = (*decoder)(temp, sequences_metadata);
    }
  } else if ((model_config.model_name == "deepseekV3")) {
    for (int layer = 0; layer < model_config.first_k_dense; layer++) {
      decoder = get_module("decoder_" + std::to_string(layer));
      out = (*decoder)(temp, sequences_metadata);
    }
    for (int layer = model_config.first_k_dense; layer < model_config.num_layers; layer++) {
      decoder = get_module("MoE_decoder_" + std::to_string(layer));
      out = (*decoder)(temp, sequences_metadata);
    }
  } else {
    fail("No configuration of " + model_config.model_name);
  }
  Module::Ptr lm_head = get_module("lm_head");
  out = (*lm_head)(out, sequences_metadata);
  return out;
}
};  // namespace llm_system
