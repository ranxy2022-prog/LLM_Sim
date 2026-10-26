// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/parallel.h"

#include "common/assert.h"
#include "module/communication.h"
// ColumnParallelLinear //

namespace llm_system {

/// @brief 初始化 ColumnParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
ColumnParallelLinear::ColumnParallelLinear(std::string& prefix,
                                           std::string& name, int input_size,
                                           int output_size,
                                           std::vector<int> device_list,
                                           Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int parallel_num = device_list.size();
  assertTrue(output_size % parallel_num == 0,
             "output_size mod parallel_num != 0");

  Module::Ptr linear =
      Linear::Create(module_map_name, "Linear", input_size,
                     output_size / parallel_num, device_list, device);
  add_module(linear);
}

/// @brief 调用本设备的列分片线性子模块，返回局部输出形状。
Tensor::Ptr ColumnParallelLinear::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr linear = get_module("Linear");

  Tensor::Ptr output_tensor = (*linear)(input, sequences_metadata);
  return output_tensor;
}

// RowParallelLinear //

/// @brief 初始化 RowParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
RowParallelLinear::RowParallelLinear(std::string& prefix, std::string& name,
                                     int input_size, int output_size,
                                     std::vector<int> device_list,
                                     Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  int parallel_num = device_list.size();
  assertTrue(output_size % parallel_num == 0,
             "output_size mod parallel_num != 0");

  // std::vector<int> shape = {input_size / parallel_num, output_size};
  Module::Ptr linear =
      Linear::Create(module_map_name, "Linear", input_size / parallel_num,
                     output_size, device_list, device);
  add_module(linear);
}

/// @brief 调用本设备的行分片线性子模块，返回局部计算结果描述。
Tensor::Ptr RowParallelLinear::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr linear = get_module("Linear");

  Tensor::Ptr output_tensor = (*linear)(input, sequences_metadata);
  return output_tensor;
}

/// @brief 初始化 BatchedColumnParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
BatchedColumnParallelLinear::BatchedColumnParallelLinear(std::string& prefix,
    std::string& name, int input_size, int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list),
  duplicated_input(duplicated_input),
  use_plain_linear(use_plain_linear) {
  int parallel_num = device_list.size();
  assertTrue(output_size % parallel_num == 0, "output_size mod parallel_num != 0");
  num_heads_per_device = num_heads / parallel_num;

  if(!use_plain_linear){
    Module::Ptr batched_linear =
        BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size,
          output_size / num_heads, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
  else{
    Module::Ptr batched_linear =
    BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size,
      output_size / parallel_num, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
}

/// @brief 为本设备负责的注意力头调用列并行线性子模块，组织输出张量列表。
TensorVec BatchedColumnParallelLinear::forward(
  const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
    
  Module::Ptr batched_linear = get_module("Batched_Linear");
  TensorVec output_vec = (*batched_linear)(input, sequences_metadata);

  return output_vec;
}

// BatchedRowParallelLinear //

/// @brief 初始化 BatchedRowParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
BatchedRowParallelLinear::BatchedRowParallelLinear(std::string& prefix, std::string& name,
    int input_size, int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list, Device::Ptr device)
  : Module(prefix, name, device, device_list),
  duplicated_input(duplicated_input),
  use_plain_linear(use_plain_linear) {
  int parallel_num = device_list.size();
  assertTrue(input_size % parallel_num == 0, "input_size mod parallel_num != 0");
  num_heads_per_device = num_heads / parallel_num;

  if(!use_plain_linear){
    Module::Ptr batched_linear =
        BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size / num_heads,
          output_size, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
  else{
    Module::Ptr batched_linear =
    BatchedLinear::Create(module_map_name, "Batched_Linear", num_heads_per_device, input_size / parallel_num,
      output_size, duplicated_input, use_plain_linear, device_list, device);
    add_module(batched_linear);
  }
}

/// @brief 为本设备负责的注意力头调用行并行线性子模块，组织局部输出列表。
TensorVec BatchedRowParallelLinear::forward(
    const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
    
      
    Module::Ptr batched_linear = get_module("Batched_Linear");
    TensorVec output_vec = (*batched_linear)(input, sequences_metadata);

    return output_vec;
}

// SelfAttentionParallel //

/// @brief 初始化 SelfAttentionParallel 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
SelfAttentionParallel::SelfAttentionParallel(std::string& prefix,
                                             std::string& name, int head_dim,
                                             int num_heads, int num_kv_heads,
                                             int max_seq_len, int batch_size, int qk_rope_head_dim,
                                             bool compressed_kv,
                                             std::vector<int> device_list,
                                             Device::Ptr device)
    : Module(prefix, name, device, device_list),
      head_dim(head_dim),
      num_heads(num_heads),
      num_kv_heads(num_kv_heads),
      qk_rope_head_dim(qk_rope_head_dim),
      compressed_kv(compressed_kv) {
  int parallel_num = device_list.size();

  assertTrue(num_heads % parallel_num == 0, "num_head mod parallel_num == 0");
  assertTrue(num_kv_heads % parallel_num == 0,
             "num_kv_head mod parallel_num == 0");

  Module::Ptr attention_split = AttentionSplit::Create(
  module_map_name, "AttentionSplit", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, 0, false, device_list,
  device);
  add_module(attention_split);

  Module::Ptr attention_sum = SelfAttentionSum::Create(
  module_map_name, "AttentionSum", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, device_list,
  device);
  add_module(attention_sum);

  Module::Ptr attention_gen = SelfAttentionGen::Create(
  module_map_name, "AttentionGen", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, device_list,
  device);
  add_module(attention_gen);

  Module::Ptr attention_merge = AttentionMerge::Create(
  module_map_name, "AttentionMerge", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, 0, false, device_list,
  device);
  add_module(attention_merge);

  // auto attn_sync = Sync::Create(module_map_name, "AttnSync",
  //                               {device->device_total_rank}, device);
  // add_module(attn_sync);
}

/// @brief 调用本设备的注意力拆分、prefill/decode 计算和合并子模块，返回并行分片输出。
Tensor::Ptr SelfAttentionParallel::forward(
    const Tensor::Ptr input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attention_split = get_module("AttentionSplit");
  Module::Ptr attention_gen = get_module("AttentionGen");
  Module::Ptr attention_sum = get_module("AttentionSum");
  Module::Ptr attention_merge = get_module("AttentionMerge");

  // Module::Ptr attention_sync = get_module("AttnSync");

  TensorVec tensor_vec;
  tensor_vec.resize(0);
  tensor_vec.push_back(input);
  tensor_vec = (*attention_split)(tensor_vec, sequences_metadata);

  Tensor::Ptr sum_out = (*attention_sum)(tensor_vec[0], sequences_metadata);
  Tensor::Ptr gen_out = (*attention_gen)(tensor_vec[1], sequences_metadata);

  //(*attention_sync)(input, sequences_metadata);

  tensor_vec.resize(0);
  tensor_vec.push_back(sum_out);
  tensor_vec.push_back(gen_out);

  TensorVec output_tensor = (*attention_merge)(tensor_vec, sequences_metadata);
  // Tensor::Ptr output_tensor_out = (*attention_sum)(input,
  // sequences_metadata); Tensor::Ptr output_tensor = (*attention_gen)(input,
  // sequences_metadata);

  return output_tensor.at(0);
}

/// @brief 初始化 MultiLatentAttentionParallel 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
MultiLatentAttentionParallel::MultiLatentAttentionParallel(std::string& prefix,
    std::string& name, int head_dim,
    int num_heads, int num_kv_heads,
    int max_seq_len, int batch_size, int qk_rope_head_dim,
    bool compressed_kv, bool use_flash_mla, bool use_flash_attention,
    std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  compressed_kv(compressed_kv) {
  int parallel_num = device_list.size();

  assertTrue(num_heads % parallel_num == 0, "num_head mod parallel_num == 0");

  Module::Ptr attention_split = AttentionSplit::Create(
  module_map_name, "AttentionSplit", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, 0, false, device_list,
  device);
  add_module(attention_split);

  Module::Ptr attention_sum = MultiLatentAttentionSum::Create(
  module_map_name, "AttentionSum", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, use_flash_attention,
  device_list, device);
  add_module(attention_sum);

  Module::Ptr attention_gen = MultiLatentAttentionGen::Create(
  module_map_name, "AttentionGen", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, compressed_kv, use_flash_mla, device_list,
  device);
  add_module(attention_gen);

  Module::Ptr attention_merge = AttentionMerge::Create(
  module_map_name, "AttentionMerge", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, 0, false,  device_list,
  device);
  add_module(attention_merge);
}

/// @brief 在非吸收 MLA 路径中组织本设备的 prefill/decode 注意力及输出形状。
TensorVec MultiLatentAttentionParallel::forward(
  const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attention_split = get_module("AttentionSplit");
  Module::Ptr attention_gen = get_module("AttentionGen");
  Module::Ptr attention_sum = get_module("AttentionSum");
  Module::Ptr attention_merge = get_module("AttentionMerge");

  // Module::Ptr attention_sync = get_module("AttnSync");
  
  TensorVec tensor_vec;
  tensor_vec.push_back(input[0]);
  tensor_vec = (*attention_split)(tensor_vec, sequences_metadata);

  Tensor::Ptr sum_out = (*attention_sum)(tensor_vec[0], sequences_metadata);
  Tensor::Ptr gen_out = (*attention_gen)(tensor_vec[1], sequences_metadata);

  // //(*attention_sync)(input, sequences_metadata);

  tensor_vec.resize(0);
  tensor_vec.push_back(sum_out);
  tensor_vec.push_back(gen_out);

  TensorVec output_tensor = (*attention_merge)(tensor_vec, sequences_metadata);

  return output_tensor;
}
/// @brief 初始化 AbsorbMLAParallel 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AbsorbMLAParallel::AbsorbMLAParallel(std::string& prefix,
    std::string& name, int head_dim,
    int num_heads, int num_kv_heads,
    int max_seq_len, int batch_size, int qk_rope_head_dim,
    int kv_lora_rank, bool compressed_kv, bool use_flash_mla,
    std::vector<int> device_list,
    Device::Ptr device)
  : Module(prefix, name, device, device_list),
  head_dim(head_dim),
  num_heads(num_heads),
  num_kv_heads(num_kv_heads),
  qk_rope_head_dim(qk_rope_head_dim),
  compressed_kv(compressed_kv) {
  int parallel_num = device_list.size();

  assertTrue(num_heads % parallel_num == 0, "num_head mod parallel_num == 0");

  Module::Ptr attention_split = AttentionSplit::Create(
  module_map_name, "AttentionSplit", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, true, device_list,
  device);
  add_module(attention_split);

  Module::Ptr attention_sum = AbsorbMLASum::Create(
  module_map_name, "AttentionSum", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, device_list,
  device);
  add_module(attention_sum);

  Module::Ptr attention_gen = AbsorbMLAGen::Create(
  module_map_name, "AttentionGen", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, compressed_kv, 
  use_flash_mla, device_list, device);
  add_module(attention_gen);

  Module::Ptr attention_merge = AttentionMerge::Create(
  module_map_name, "AttentionMerge", head_dim, num_heads / parallel_num,
  num_kv_heads / parallel_num, max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, true, device_list,
  device);
  add_module(attention_merge);
}

/// @brief 在吸收 MLA 路径中组织各头的 prefill/decode 注意力与张量列表输出。
TensorVec AbsorbMLAParallel::forward(
  const TensorVec input, BatchedSequence::Ptr sequences_metadata) {
  Module::Ptr attention_split = get_module("AttentionSplit");
  Module::Ptr attention_gen = get_module("AttentionGen");
  Module::Ptr attention_sum = get_module("AttentionSum");
  Module::Ptr attention_merge = get_module("AttentionMerge");

  // TODO: device가 처리해야하는 head갯수만큼, score (input (S,512) x c_kv (512, S))

  // input: (num head per device, S, 512)
  TensorVec tensor_vec = (*attention_split)(input, sequences_metadata);
  
  int num_heads_per_device = tensor_vec.size() / 2;

  Tensor::Ptr sum_tensor = tensor_vec[0];
  Tensor::Ptr gen_tensor = tensor_vec[num_heads_per_device];
  
  Tensor::Ptr sum_out = (*attention_sum)(sum_tensor, sequences_metadata);
  Tensor::Ptr gen_out = (*attention_gen)(gen_tensor, sequences_metadata);
  
  tensor_vec.resize(0);
  tensor_vec.push_back(sum_out);
  tensor_vec.push_back(gen_out);

  TensorVec output_tensor = (*attention_merge)(tensor_vec, sequences_metadata);
  
  return output_tensor;
}
}  // namespace llm_system