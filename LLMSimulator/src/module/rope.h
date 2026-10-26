// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 旋转位置编码性能模块：维护 sin/cos 形状，估计逐元素操作与访存开销。
class RoPE : public Module {
    public:
        using Ptr = std::shared_ptr<RoPE>;

        /// @brief RoPE 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
        [[nodiscard]] static Ptr Create(std::string prefix, std::string name, int num_heads,
                                        int qk_rope_head_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new RoPE(prefix, name, num_heads, qk_rope_head_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        /// @brief 初始化 RoPE 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
        RoPE(std::string& prefix, std::string& name, int num_heads, int qk_rope_head_dim,
                    std::vector<int> device_list, Device::Ptr device);
        /// @brief RoPE 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
        RoPE() = default;

        /// @brief 构造 sin/cos 和输出形状，按元素运算/访存量估算位置编码耗时并更新设备时间。
        Tensor::Ptr forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) override;
        int qk_rope_head_dim;
};
/// @brief 多头/多张量 RoPE 性能模块：汇总形状估算成本，返回原输入张量列表。
class BatchedRoPE : public Module {
    public:
        using Ptr = std::shared_ptr<BatchedRoPE>;

        /// @brief BatchedRoPE 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
        [[nodiscard]] static Ptr Create(std::string prefix, std::string name, int num_heads,
                                        int qk_rope_head_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new BatchedRoPE(prefix, name, num_heads, qk_rope_head_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        /// @brief 初始化 BatchedRoPE 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
        BatchedRoPE(std::string& prefix, std::string& name, int num_heads, int qk_rope_head_dim,
                    std::vector<int> device_list, Device::Ptr device);
        /// @brief BatchedRoPE 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
        BatchedRoPE() = default;

        /// @brief 按多张量总元素数估算 RoPE 耗时并更新设备时间，当前返回原输入列表。
        TensorVec forward(const TensorVec input,
                            BatchedSequence::Ptr sequences_metadata) override;
        int qk_rope_head_dim;
    };
}  // namespace llm_system