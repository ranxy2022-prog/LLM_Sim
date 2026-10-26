// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。

#include "module/attention.h"

#include "common/assert.h"
#include "hardware/hardware_config.h"
#include "module/base.h"

namespace llm_system {

// SelfAttentionGen //

/// @brief 初始化 SelfAttentionGen 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
SelfAttentionGen::SelfAttentionGen(std::string& prefix, std::string& name,
                                   int head_dim, int num_heads,
                                   int num_kv_heads, int max_seq_len,
                                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                                   Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {max_seq_len, head_dim};
  for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
    for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
      Tensor::Ptr k_cache = Tensor::Create(
          "k_cache_" + std::to_string(seq_idx) + "_" + std::to_string(kv_idx),
          shape, "cache", device, device->model_config.precision_byte);
      add_tensor(k_cache);

      Tensor::Ptr v_cache = Tensor::Create(
          "v_cache_" + std::to_string(seq_idx) + "_" + std::to_string(kv_idx),
          shape, "cache", device, device->model_config.precision_byte);
      add_tensor(v_cache);
    }
  }

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 常规 decode 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr SelfAttentionGen::forward(const Tensor::Ptr input,
                                      BatchedSequence::Ptr sequences_metadata) {
  std::vector<int> shape = {input->shape[0], num_heads * head_dim};

  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.low_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformLow();
  }

  layer_info.attention_group_size = num_heads / num_kv_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);
  tensor_list.push_back(get_cache("k_cache", 0, 0, false));
  tensor_list.push_back(get_cache("v_cache", 0, 0, false));

  device->execution(LayerType::ATTENTION_GEN, tensor_list, sequences_metadata,
                    layer_info);

  return output_tensor;
}

// SelfAttentionSum //
/// @brief 初始化 SelfAttentionSum 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
SelfAttentionSum::SelfAttentionSum(std::string& prefix, std::string& name,
                                   int head_dim, int num_heads,
                                   int num_kv_heads, int max_seq_len,
                                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                                   Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {batch_size, max_seq_len, num_kv_heads, head_dim};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 常规 prefill 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr SelfAttentionSum::forward(const Tensor::Ptr input,
                                      BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  std::vector<int> shape = {input->shape[0], num_heads * head_dim};
  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.high_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformHigh();
  }

  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  tensor_list.push_back(output_tensor);

  device->execution(LayerType::ATTENTION_SUM, tensor_list, sequences_metadata,
                    layer_info);

  return output_tensor;
}

// SelfAttentionMixed //
/// @brief 初始化 SelfAttentionMixed 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
SelfAttentionMixed::SelfAttentionMixed(std::string& prefix, std::string& name,
                                       int head_dim, int num_heads,
                                       int num_kv_heads, int max_seq_len,
                                       int batch_size, int qk_rope_head_dim,
                                       std::vector<int> device_list,
                                       Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {batch_size, max_seq_len, num_kv_heads, head_dim};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 常规混合阶段 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr SelfAttentionMixed::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);
  tensor_list.push_back(input);

  std::vector<int> shape = {input->shape[0], num_heads * head_dim};

  device->execution(LayerType::ATTENTION_MIXED, tensor_list, sequences_metadata,
                    layer_info);

  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  return output_tensor;
}

// AttentionSplit //
/// @brief 初始化 AttentionSplit 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AttentionSplit::AttentionSplit(std::string& prefix, std::string& name,
                               int head_dim, int num_heads, int num_kv_heads,
                               int max_seq_len, int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                               bool use_absorb, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim),
      kv_lora_rank(kv_lora_rank),
      use_absorb(use_absorb) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {1, 1};

  Tensor::Ptr sum_tensor = Tensor::Create("sum_tensor", shape, "act", device, device->model_config.precision_byte);
  add_tensor(sum_tensor);
  Tensor::Ptr gen_tensor = Tensor::Create("gen_tensor", shape, "act", device, device->model_config.precision_byte);
  add_tensor(gen_tensor);
}

/// @brief 根据 batch 中两阶段的 token 数拆分注意力输入描述，不进行真实数据复制。
TensorVec AttentionSplit::forward(const TensorVec input,
                                  BatchedSequence::Ptr sequences_metadata) {
  int sum_token = sequences_metadata->get_sum_process_token();
  int gen_token = sequences_metadata->get_gen_process_token();

  TensorVec tensor_list;
  std::vector<int> sum_shape = {1, 1};
  std::vector<int> gen_shape = {1, 1};

  if(qk_rope_head_dim == 0){
    sum_shape = {sum_token, (num_heads + num_kv_heads * 2) * head_dim};
    gen_shape = {gen_token, (num_heads + num_kv_heads * 2) * head_dim};
  }
  else{
    if(use_absorb){
      int num_heads_per_device = input.size();
      sum_shape = {num_heads_per_device, sum_token, kv_lora_rank};
      gen_shape = {num_heads_per_device, gen_token, kv_lora_rank};
    }
    else{
      sum_shape = {sum_token, num_heads * (2 * (head_dim + qk_rope_head_dim) + head_dim )};
      gen_shape = {gen_token, num_heads * (2 * (head_dim + qk_rope_head_dim) + head_dim )};
    }
  }
  Tensor::Ptr sum_tensor = get_activation("sum_tensor", sum_shape);
  Tensor::Ptr gen_tensor = get_activation("gen_tensor", gen_shape);

  device->status.device_time =
      std::max(device->status.device_time,
              std::max(device->status.low_time, device->status.high_time));
  device->status.high_time = device->status.device_time;
  device->status.low_time = device->status.device_time;

  if (device->config.parallel_execution) {
    sum_tensor->setPerformHigh();
    gen_tensor->setPerformLow();
  }

  tensor_list.resize(0);
  if(use_absorb){
    tensor_list.push_back(sum_tensor);
    tensor_list.push_back(gen_tensor);
  }
  else{
    tensor_list.push_back(sum_tensor);
    tensor_list.push_back(gen_tensor);
  }
  return tensor_list;
}

// AttentionMerge //
/// @brief 初始化 AttentionMerge 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AttentionMerge::AttentionMerge(std::string& prefix, std::string& name,
                               int head_dim, int num_heads, int num_kv_heads,
                               int max_seq_len, int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                               bool use_absorb, std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      kv_lora_rank(kv_lora_rank),
      use_absorb(use_absorb) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {1, 1};

  Tensor::Ptr output_tensor =
      Tensor::Create("output_tensor", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output_tensor);
}

/// @brief 按两阶段结果构造统一输出形状，供后续投影使用。
TensorVec AttentionMerge::forward(const TensorVec input,
                                  BatchedSequence::Ptr sequences_metadata) {
  int sum_token = sequences_metadata->get_sum_process_token();
  int gen_token = sequences_metadata->get_gen_process_token();

  if(use_absorb){

    std::vector<int> shape = {sum_token + gen_token, kv_lora_rank};
    Tensor::Ptr output_tensor = get_activation("output_tensor", shape);

    device->status.device_time =
        std::max(device->status.device_time,
                std::max(device->status.low_time, device->status.high_time));
    device->status.high_time = device->status.device_time;
    device->status.low_time = device->status.device_time;

    TensorVec tensor_list;
    tensor_list.resize(0);
    for(int i = 0 ; i < num_heads; i ++){
      tensor_list.push_back(output_tensor);
    }

    return tensor_list;
  }
  else{
    assertTrue(input[0]->dim(0) == sum_token,
              "Dimenison of input tensor of AttentionMerge is not matched");
    assertTrue(input[1]->dim(0) == gen_token,
              "Dimenison of input tensor of AttentionMerge is not matched");

    std::vector<int> shape = {sum_token + gen_token, num_heads * head_dim};
    Tensor::Ptr output_tensor = get_activation("output_tensor", shape);

    device->status.device_time =
        std::max(device->status.device_time,
                std::max(device->status.low_time, device->status.high_time));
    device->status.high_time = device->status.device_time;
    device->status.low_time = device->status.device_time;

    TensorVec tensor_list;
    tensor_list.resize(0);
    tensor_list.push_back(output_tensor);

    return tensor_list;
  }
}

// MultiLatentAttentionGen //

/// @brief 初始化 MultiLatentAttentionGen 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
MultiLatentAttentionGen::MultiLatentAttentionGen(std::string& prefix, std::string& name,
    int head_dim, int num_heads,
    int num_kv_heads, int max_seq_len,
    int batch_size, int qk_rope_head_dim, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  compressed_kv(compressed_kv),
  use_flash_mla(use_flash_mla) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {max_seq_len, head_dim};
  if(!compressed_kv){
    for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
      for (int kv_idx = 0; kv_idx < num_kv_heads; kv_idx++) {
        Tensor::Ptr k_cache = Tensor::Create(
          "k_cache_" + std::to_string(seq_idx) + "_" + std::to_string(kv_idx),
          shape, "cache", device, device->model_config.precision_byte);
          add_tensor(k_cache);
    
        Tensor::Ptr v_cache = Tensor::Create(
          "v_cache_" + std::to_string(seq_idx) + "_" + std::to_string(kv_idx),
          shape, "cache", device, device->model_config.precision_byte);
          add_tensor(v_cache);
      }
    }
  }

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 非吸收 MLA decode 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr MultiLatentAttentionGen::forward(const Tensor::Ptr input,
      BatchedSequence::Ptr sequences_metadata) {
  std::vector<int> shape = {input->shape[0], num_heads * head_dim};

  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.low_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformLow();
  }else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.num_heads = num_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.use_flash_mla = use_flash_mla;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);
  if(compressed_kv){
    
  }
  else{
    tensor_list.push_back(get_cache("k_cache", 0, 0, false));
    tensor_list.push_back(get_cache("v_cache", 0, 0, false));
  }

  device->execution(LayerType::MLA_GEN, tensor_list, sequences_metadata, layer_info);

  return output_tensor;
}

// MultiLatentAttentionSum //
/// @brief 初始化 MultiLatentAttentionSum 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
MultiLatentAttentionSum::MultiLatentAttentionSum(std::string& prefix, std::string& name,
  int head_dim, int num_heads,
  int num_kv_heads, int max_seq_len,
  int batch_size, int qk_rope_head_dim, bool use_flash_attention, std::vector<int> device_list,
  Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  use_flash_attention(use_flash_attention) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {batch_size, max_seq_len, num_kv_heads, head_dim};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 非吸收 MLA prefill 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr MultiLatentAttentionSum::forward(const Tensor::Ptr input,
     BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  std::vector<int> shape = input->shape;
  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  if (input->parallel_execution) {
    layer_info.parallel_execution = true;
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }
  else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.use_flash_attention = use_flash_attention;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  tensor_list.push_back(output_tensor);

  device->execution(LayerType::MLA_SUM, tensor_list, sequences_metadata,
  layer_info);

  return output_tensor;
}

// MultiLatentAttentionGen //

/// @brief 初始化 AbsorbMLAGen 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AbsorbMLAGen::AbsorbMLAGen(std::string& prefix, std::string& name,
    int head_dim, int num_heads,
    int num_kv_heads, int max_seq_len,
    int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool compressed_kv, 
    bool use_flash_mla, std::vector<int> device_list, Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  kv_lora_rank(kv_lora_rank),
  compressed_kv(compressed_kv),
  use_flash_mla(use_flash_mla) {
  int parallel_num = device_list.size();

  std::vector<int> latent_kv_shape = {max_seq_len, kv_lora_rank};
  std::vector<int> latent_pe_shape = {max_seq_len, qk_rope_head_dim};
  std::vector<int> out_shape = {num_kv_heads / parallel_num, max_seq_len, kv_lora_rank};

  for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
    Tensor::Ptr latent_kv_cache = Tensor::Create(
      "latent_kv_cache_" + std::to_string(seq_idx),
      latent_kv_shape, "cache", device, device->model_config.precision_byte);
      add_tensor(latent_kv_cache);

    Tensor::Ptr latent_pe_cache = Tensor::Create(
      "latent_pe_cache_" + std::to_string(seq_idx),
      latent_pe_shape, "cache", device, device->model_config.precision_byte);
      add_tensor(latent_pe_cache);
  }

  Tensor::Ptr output = Tensor::Create("attn_output", out_shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 吸收 MLA decode 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr AbsorbMLAGen::forward(const Tensor::Ptr input,
      BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output_tensor = get_activation("attn_output", input->shape);

  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.high_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformHigh();
  } else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.num_heads = num_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.kv_lora_rank = kv_lora_rank;
  layer_info.use_flash_mla = use_flash_mla;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  int batch_size = sequences_metadata->get_gen().size();
  
  for (int seq_idx = 0; seq_idx < batch_size; seq_idx++) {
    Tensor::Ptr latent_kv_cache = get_cache("latent_kv_cache", seq_idx, 0, true);
    Tensor::Ptr latent_pe_cache = get_cache("latent_pe_cache", seq_idx, 0, true);
  }

  device->execution(LayerType::ABSORBED_MLA_GEN, tensor_list, sequences_metadata, layer_info);
  
  return output_tensor;
}

// MultiLatentAttentionSum //
/// @brief 初始化 AbsorbMLASum 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AbsorbMLASum::AbsorbMLASum(std::string& prefix, std::string& name,
  int head_dim, int num_heads,
  int num_kv_heads, int max_seq_len,
  int batch_size, int qk_rope_head_dim, int kv_lora_rank, std::vector<int> device_list,
  Device::Ptr device)
  : Module(prefix, name, device, device_list, true),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  kv_lora_rank(kv_lora_rank) {
  int parallel_num = device_list.size();

  std::vector<int> shape = {num_kv_heads / parallel_num, max_seq_len, kv_lora_rank};

  Tensor::Ptr output = Tensor::Create("attn_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 根据 batch 的 token/上下文长度设置 吸收 MLA prefill 张量和缓存描述，提交对应注意力性能估算并返回输出。
Tensor::Ptr AbsorbMLASum::forward(const Tensor::Ptr input,
     BatchedSequence::Ptr sequences_metadata) {
  LayerInfo layer_info;
  layer_info.processor_type = device->config.processor_type;
  std::vector<int> shape = input->shape;
  Tensor::Ptr output_tensor = get_activation("attn_output", shape);

  if (input->parallel_execution) {
    layer_info.processor_type = {device->config.high_processor_type};
    layer_info.parallel_execution = true;
    output_tensor->setPerformHigh();
  } else if(device->config.processor_type.size() != 1){
    layer_info.processor_type = {device->config.high_processor_type};
    output_tensor->setPerformHigh();
  }

  layer_info.attention_group_size = head_dim / num_heads;
  layer_info.num_heads = num_heads;
  layer_info.num_kv_heads = num_kv_heads;
  layer_info.head_dim = head_dim;
  layer_info.qk_rope_head_dim = qk_rope_head_dim;
  layer_info.kv_lora_rank = kv_lora_rank;

  std::vector<Tensor::Ptr> tensor_list;
  tensor_list.resize(0);
  tensor_list.push_back(input);

  tensor_list.push_back(output_tensor);

  device->execution(LayerType::ABSORBED_MLA_SUM, tensor_list, sequences_metadata,
  layer_info);

  
  return output_tensor;
}

}  // namespace llm_system