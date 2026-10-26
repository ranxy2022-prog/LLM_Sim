// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 残差加法性能模块：按一次逐元素加法与两读一写估计耗时，不计算实际张量值。
class Residual : public Module {
    public:
        using Ptr = std::shared_ptr<Residual>;

        /// @brief Residual 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
        [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                        int hidden_dim, std::vector<int> device_list,
                                        Device::Ptr device) {
            Ptr ptr = Ptr(new Residual(prefix, name, hidden_dim, device_list, device));
            ptr->set_tensor_module();
            return ptr;
        };

    private:
        /// @brief 初始化 Residual 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
        Residual(std::string& prefix, std::string& name, int hidden_dim,
                    std::vector<int> device_list, Device::Ptr device);
        /// @brief Residual 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
        Residual() = default;

        /// @brief 按 M*K 加法、三份张量访存量估算耗时，更新设备及必要的高/低时间线。
        Tensor::Ptr forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) override;
    };
}  // namespace llm_system