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

/// @brief 非吸收 MLA 的 KV 恢复描述模块：维护压缩缓存及待上投影张量形状，本身不进行数值解压。
class CompressedKVRestore : public Module {
    public:
     using Ptr = std::shared_ptr<CompressedKVRestore>;
     /// @brief CompressedKVRestore 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
     [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                     int head_dim, int num_heads, int max_seq_len, int batch_size, int kv_lora_rank, int qk_rope_head_dim,
                                     bool compressed_kv, std::vector<int> device_list,
                                     Device::Ptr device) {
       Ptr ptr =
           Ptr(new CompressedKVRestore(prefix, name, head_dim, num_heads, 
                                  max_seq_len, batch_size, kv_lora_rank, qk_rope_head_dim, compressed_kv, device_list, device));
       ptr->set_tensor_module();
       return ptr;
     };
   
    private:
     /// @brief 初始化 CompressedKVRestore 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
     CompressedKVRestore(std::string& prefix, std::string& name, int head_dim,
                    int num_heads, int max_seq_len,
                    int batch_size, int kv_lora_rank, int qk_rope_head_dim, bool compressed_kv, std::vector<int> device_list,
                    Device::Ptr device);
     /// @brief CompressedKVRestore 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
     CompressedKVRestore() = default;
   
     /// @brief 开启压缩时按 prefill token + decode 历史长度 + decode 请求数构造潜变量/K-rope 形状；否则沿用输入形状。本身不计解压算力，上投影由后续模块承担。
     TensorVec forward(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata) override;
   
    private:
     int rank;
     int head_dim;
     int num_heads;
     int kv_lora_rank;
     int qk_rope_head_dim;
     int max_seq_len;
     bool compressed_kv;
   };

}  // namespace llm_system