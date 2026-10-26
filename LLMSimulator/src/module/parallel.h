// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <cassert>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "module/attention.h"
#include "module/linear.h"
#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 列并行线性层：按 TP 组拆分权重输出维度 N，每台设备计算部分输出特征。
class ColumnParallelLinear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<ColumnParallelLinear>;

  /// @brief ColumnParallelLinear 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new ColumnParallelLinear(prefix, name, input_size,
                                           output_size, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 ColumnParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  ColumnParallelLinear(std::string& prefix, std::string& name, int input_size,
                       int output_size, std::vector<int> device_list,
                       Device::Ptr device);
  /// @brief ColumnParallelLinear 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  ColumnParallelLinear() = default;

  /// @brief 调用本设备的列分片线性子模块，返回局部输出形状。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 行并行线性层：按 TP 组拆分输入维度 K，每台设备计算局部结果；归约由组合层组织。
class RowParallelLinear : public Module {
  // Y = XA +b, X split in column, A split in row
 public:
  using Ptr = std::shared_ptr<RowParallelLinear>;
  /// @brief RowParallelLinear 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new RowParallelLinear(prefix, name, input_size, output_size,
                                        device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 RowParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  RowParallelLinear(std::string& prefix, std::string& name, int input_size,
                    int output_size, std::vector<int> device_list,
                    Device::Ptr device);
  /// @brief RowParallelLinear 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  RowParallelLinear() = default;

  /// @brief 调用本设备的行分片线性子模块，返回局部计算结果描述。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 按注意力头组织的批量列并行线性层，管理本设备头范围和对应子矩阵乘模块。
class BatchedColumnParallelLinear : public Module {
  // Y = XA +b, split A in column
 public:
  using Ptr = std::shared_ptr<BatchedColumnParallelLinear>;

  /// @brief BatchedColumnParallelLinear 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads, 
                                  bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new BatchedColumnParallelLinear(prefix, name, input_size, output_size,
                                  num_heads, duplicated_input, use_plain_linear, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 BatchedColumnParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  BatchedColumnParallelLinear(std::string& prefix, std::string& name, int input_size,
                       int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                       Device::Ptr device);
  /// @brief BatchedColumnParallelLinear 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  BatchedColumnParallelLinear() = default;

  /// @brief 为本设备负责的注意力头调用列并行线性子模块，组织输出张量列表。
  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
  bool duplicated_input;
  bool use_plain_linear;
};

/// @brief 按注意力头组织的批量行并行线性层，管理本设备头范围和局部结果。
class BatchedRowParallelLinear : public Module {
  // Y = XA +b, X split in column, A split in row
 public:
  using Ptr = std::shared_ptr<BatchedRowParallelLinear>;
  /// @brief BatchedRowParallelLinear 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads,
                                  bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new BatchedRowParallelLinear(prefix, name, input_size, output_size, num_heads, 
                                        duplicated_input, use_plain_linear, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 BatchedRowParallelLinear 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  BatchedRowParallelLinear(std::string& prefix, std::string& name, int input_size,
                    int output_size, int num_heads, bool duplicated_input, bool use_plain_linear, std::vector<int> device_list,
                    Device::Ptr device);
  /// @brief BatchedRowParallelLinear 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  BatchedRowParallelLinear() = default;

  /// @brief 为本设备负责的注意力头调用行并行线性子模块，组织局部输出列表。
  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
  bool duplicated_input;
  bool use_plain_linear;
};

/// @brief 常规注意力的并行封装：按设备划分注意力头，组织 prefill/decode 子模块。
class SelfAttentionParallel : public Module {
 public:
  using Ptr = std::shared_ptr<SelfAttentionParallel>;
  /// @brief SelfAttentionParallel 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int head_dim, int num_heads, int num_kv_heads,
                                  int max_seq_len, int batch_size, int qk_rope_head_dim, bool compressed_kv,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new SelfAttentionParallel(prefix, name, head_dim, num_heads,
                                            num_kv_heads, max_seq_len,
                                            batch_size, qk_rope_head_dim, compressed_kv, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 SelfAttentionParallel 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  SelfAttentionParallel(std::string& prefix, std::string& name, int head_dim,
                        int num_heads, int num_kv_heads, int max_seq_len, int qk_rope_head_dim,
                        int batch_size, bool compressed_kv, std::vector<int> device_list,
                        Device::Ptr device);
  /// @brief SelfAttentionParallel 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  SelfAttentionParallel() = default;

  /// @brief 调用本设备的注意力拆分、prefill/decode 计算和合并子模块，返回并行分片输出。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;

 private:
  int rank;
  int head_dim;
  int num_heads;
  int num_kv_heads;
  int qk_rope_head_dim;
  bool compressed_kv;
};

/// @brief 预留的行方向张量分片模块；当前只有类接口，带参数构造和 forward 未提供定义。
class RowSplit : public Module {
  // Split tensor in row
 public:
  using Ptr = std::shared_ptr<RowSplit>;
  /// @brief RowSplit 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new RowSplit(prefix, name, input_size, output_size, num_heads, 
                                        device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief RowSplit 的带参数构造声明；当前自有源码未提供对应定义，是尚未完整实现的分片接口。
  RowSplit(std::string& prefix, std::string& name, int input_size,
                    int output_size, int num_heads, std::vector<int> device_list,
                    Device::Ptr device);
  /// @brief RowSplit 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  RowSplit() = default;

  /// @brief 预留的张量分片 forward 接口；当前自有源码未提供定义，不能据此认为该模块可完整执行。
  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
};

/// @brief 预留的列方向张量分片模块；当前只有类接口，带参数构造和 forward 未提供定义。
class ColumnSplit : public Module {
  // Split tensor in column 
 public:
  using Ptr = std::shared_ptr<ColumnSplit>;
  /// @brief ColumnSplit 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  int input_size, int output_size, int num_heads,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new ColumnSplit(prefix, name, input_size, output_size, num_heads, 
                                        device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief ColumnSplit 的带参数构造声明；当前自有源码未提供对应定义，是尚未完整实现的分片接口。
  ColumnSplit(std::string& prefix, std::string& name, int input_size,
                    int output_size, int num_heads, std::vector<int> device_list,
                    Device::Ptr device);
  /// @brief ColumnSplit 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  ColumnSplit() = default;

  /// @brief 预留的张量分片 forward 接口；当前自有源码未提供定义，不能据此认为该模块可完整执行。
  TensorVec forward(const TensorVec input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int num_heads_per_device;
  int head_offset;
};

/// @brief 非吸收 MLA 的并行封装：按设备分配头和张量，组织 prefill/decode 执行。
class MultiLatentAttentionParallel : public Module {
  public:
   using Ptr = std::shared_ptr<MultiLatentAttentionParallel>;
   /// @brief MultiLatentAttentionParallel 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, bool compressed_kv,
                                   bool use_flash_mla, bool use_flash_attention, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new MultiLatentAttentionParallel(prefix, name, head_dim, num_heads,
                                             num_kv_heads, max_seq_len,
                                             batch_size, qk_rope_head_dim, compressed_kv, use_flash_mla,
                                             use_flash_attention, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    /// @brief 初始化 MultiLatentAttentionParallel 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
    MultiLatentAttentionParallel(std::string& prefix, std::string& name, int head_dim,
                         int num_heads, int num_kv_heads, int max_seq_len, int qk_rope_head_dim,
                         int batch_size, bool compressed_kv, bool use_flash_mla, bool use_flash_attention,
                         std::vector<int> device_list, Device::Ptr device);
    /// @brief MultiLatentAttentionParallel 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
    MultiLatentAttentionParallel() = default;
 
   /// @brief 在非吸收 MLA 路径中组织本设备的 prefill/decode 注意力及输出形状。
   TensorVec forward(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
};

/// @brief 权重吸收 MLA 的多头并行封装，组织各设备的 prefill/decode 注意力路径。
class AbsorbMLAParallel : public Module {
  public:
   using Ptr = std::shared_ptr<AbsorbMLAParallel>;
   /// @brief AbsorbMLAParallel 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
   [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                   int head_dim, int num_heads, int num_kv_heads,
                                   int max_seq_len, int batch_size, int qk_rope_head_dim, int kv_lora_rank,
                                   bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                                   Device::Ptr device) {
     Ptr ptr = Ptr(new AbsorbMLAParallel(prefix, name, head_dim, num_heads,
                                             num_kv_heads, max_seq_len,
                                             batch_size, qk_rope_head_dim, kv_lora_rank, compressed_kv, use_flash_mla, device_list, device));
     ptr->set_tensor_module();
     return ptr;
   };
 
  private:
    /// @brief 初始化 AbsorbMLAParallel 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
    AbsorbMLAParallel(std::string& prefix, std::string& name, int head_dim,
                         int num_heads, int num_kv_heads, int max_seq_len, int qk_rope_head_dim,
                         int batch_size, int kv_lora_rank, bool compressed_kv, bool use_flash_mla, std::vector<int> device_list,
                         Device::Ptr device);
    /// @brief AbsorbMLAParallel 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
    AbsorbMLAParallel() = default;
 
   /// @brief 在吸收 MLA 路径中组织各头的 prefill/decode 注意力与张量列表输出。
   TensorVec forward(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata) override;
 
  private:
   int rank;
   int head_dim;
   int num_heads;
   int num_kv_heads;
   int qk_rope_head_dim;
   bool compressed_kv;
};

}  // namespace llm_system