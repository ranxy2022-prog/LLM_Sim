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

/// @brief 常规 decode 注意力模块：结合历史 KV 长度组织本轮生成阶段的算子描述。
class SelfAttentionGen : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionGen>;
  /// @brief SelfAttentionGen 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionGen(prefix, name, head_dim, num_heads,
                                       num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                       device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 SelfAttentionGen 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  SelfAttentionGen(std::string& prefix, std::string& name, int head_dim,
                   int num_heads, int num_kv_heads, int max_seq_len,
                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                   Device::Ptr device);
  /// @brief SelfAttentionGen 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  SelfAttentionGen() = default;

  /// @brief 根据 batch 的 token/上下文长度设置 常规 decode 张量和缓存描述，提交对应注意力性能估算并返回输出。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // std::vector<int> device_list;
};

/// @brief 常规 prefill 注意力模块：根据本轮输入 token 构造上下文处理阶段的算子描述。
class SelfAttentionSum : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionSum>;
  /// @brief SelfAttentionSum 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionSum(prefix, name, head_dim, num_heads,
                                       num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                       device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 SelfAttentionSum 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  SelfAttentionSum(std::string& prefix, std::string& name, int head_dim,
                   int num_heads, int num_kv_heads, int max_seq_len,
                   int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                   Device::Ptr device);
  /// @brief SelfAttentionSum 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  SelfAttentionSum() = default;

  /// @brief 根据 batch 的 token/上下文长度设置 常规 prefill 张量和缓存描述，提交对应注意力性能估算并返回输出。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // std::vector<int> device_list;
};

/// @brief 常规混合注意力模块：组织包含 prefill/decode 元数据的注意力执行请求。
class SelfAttentionMixed : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionMixed>;
  /// @brief SelfAttentionMixed 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionMixed(prefix, name, head_dim, num_heads,
                                         num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                         device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 SelfAttentionMixed 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  SelfAttentionMixed(std::string& prefix, std::string& name, int head_dim,
                     int num_heads, int num_kv_heads, int max_seq_len,
                     int batch_size, int qk_rope_head_dim, std::vector<int> device_list,
                     Device::Ptr device);
  /// @brief SelfAttentionMixed 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  SelfAttentionMixed() = default;

  /// @brief 根据 batch 的 token/上下文长度设置 常规混合阶段 张量和缓存描述，提交对应注意力性能估算并返回输出。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  // std::vector<int> device_list;
};

/// @brief 注意力输入拆分模块：按当前 batch 的 prefill/decode 组成构造阶段输入张量。
class AttentionSplit : public Module {
 public:
  using Ptr = std::shared_ptr<AttentionSplit>;
  /// @brief AttentionSplit 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim,
                                  int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new AttentionSplit(prefix, name, head_dim, num_heads, num_kv_heads,
                               max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, use_absorb, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 AttentionSplit 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  AttentionSplit(std::string& prefix, std::string& name, int head_dim,
                 int num_heads, int num_kv_heads, int max_seq_len,
                 int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                 bool use_absorb, std::vector<int> device_list,
                 Device::Ptr device);
  /// @brief AttentionSplit 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  AttentionSplit() = default;

  /// @brief 根据 batch 中两阶段的 token 数拆分注意力输入描述，不进行真实数据复制。
  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  int kv_lora_rank;
  bool use_absorb;
  // std::vector<int> device_list;
};

/// @brief 注意力输出合并模块：把阶段输出重新表示为后续模块需要的张量形状。
class AttentionMerge : public Module {
 public:
  using Ptr = std::shared_ptr<AttentionMerge>;
  /// @brief AttentionMerge 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                  int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr =
        Ptr(new AttentionMerge(prefix, name, head_dim, num_heads, num_kv_heads,
                               max_seq_len, batch_size, qk_rope_head_dim, kv_lora_rank, use_absorb, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 AttentionMerge 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  AttentionMerge(std::string& prefix, std::string& name, int head_dim,
                 int num_heads, int num_kv_heads, int max_seq_len,
                 int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool use_absorb, std::vector<int> device_list,
                 Device::Ptr device);
  /// @brief AttentionMerge 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  AttentionMerge() = default;

  /// @brief 按两阶段结果构造统一输出形状，供后续投影使用。
  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  int kv_lora_rank;
  bool use_absorb;
  // std::vector<int> device_list;
};

/// @brief 非吸收 MLA 的 decode 注意力模块，使用低秩相关维度与历史上下文组织计算。
class MultiLatentAttentionGen : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionGen>;
   /// @brief MultiLatentAttentionGen 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionGen(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        compressed_kv, use_flash_mla, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    /// @brief 初始化 MultiLatentAttentionGen 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
    MultiLatentAttentionGen(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                    Device::Ptr device);
    /// @brief MultiLatentAttentionGen 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
    MultiLatentAttentionGen() = default;
 
   /// @brief 根据 batch 的 token/上下文长度设置 非吸收 MLA decode 张量和缓存描述，提交对应注意力性能估算并返回输出。
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
   bool use_flash_mla;
   // std::vector<int> device_list;
 };
 
 /// @brief 非吸收 MLA 的 prefill 注意力模块，描述当前输入块的注意力开销。
 class MultiLatentAttentionSum : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionSum>;
   /// @brief MultiLatentAttentionSum 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                   bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionSum(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    /// @brief 初始化 MultiLatentAttentionSum 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
    MultiLatentAttentionSum(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, bool use_flash_attention, 
                    std::vector<int> device_list, Device::Ptr device);
    /// @brief MultiLatentAttentionSum 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
    MultiLatentAttentionSum() = default;
 
   /// @brief 根据 batch 的 token/上下文长度设置 非吸收 MLA prefill 张量和缓存描述，提交对应注意力性能估算并返回输出。
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool use_flash_attention;
   // std::vector<int> device_list;
 };
 
 /// @brief 权重吸收 MLA 的 decode 注意力模块，在压缩潜变量维度上描述注意力计算与缓存访问。
 class AbsorbMLAGen : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLAGen>;
   /// @brief AbsorbMLAGen 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim,
                                   int kv_lora_rank, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLAGen(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        kv_lora_rank, compressed_kv, use_flash_mla, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    /// @brief 初始化 AbsorbMLAGen 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
    AbsorbMLAGen(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, int kv_lora_rank, bool compressed_kv,
                    bool use_flash_mla, std::vector<int> device_list, Device::Ptr device);
    /// @brief AbsorbMLAGen 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
    AbsorbMLAGen() = default;
 
   /// @brief 根据 batch 的 token/上下文长度设置 吸收 MLA decode 张量和缓存描述，提交对应注意力性能估算并返回输出。
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   int kv_lora_rank;
   bool compressed_kv;
   bool use_flash_mla;
   // std::vector<int> device_list;
 };
 
 /// @brief 权重吸收 MLA 的 prefill 注意力模块，组织对应张量、缓存和硬件执行描述。
 class AbsorbMLASum : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLASum>;
   /// @brief AbsorbMLASum 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, 
                                   int kv_lora_rank, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLASum(prefix, name, head_dim, num_heads,
                                        num_kv_heads, max_seq_len, batch_size, qk_rope_head_dim,
                                        kv_lora_rank, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    /// @brief 初始化 AbsorbMLASum 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
    AbsorbMLASum(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int num_kv_heads, int max_seq_len,
                    int batch_size, int qk_rope_head_dim, int kv_lora_rank, std::vector<int> device_list,
                    Device::Ptr device);
    /// @brief AbsorbMLASum 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
    AbsorbMLASum() = default;
 
   /// @brief 根据 batch 的 token/上下文长度设置 吸收 MLA prefill 张量和缓存描述，提交对应注意力性能估算并返回输出。
   Tensor::Ptr forward(const Tensor::Ptr input,
                       BatchedSequence::Ptr sequences_metadata) override;
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   int kv_lora_rank;
   // std::vector<int> device_list;
 };
}  // namespace llm_system