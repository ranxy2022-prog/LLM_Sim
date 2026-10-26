// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include "hardware/base.h"

namespace llm_system {

/// @brief Tensor 的前向声明：此处只声明类型，完整结构见对应头文件。
class Tensor;
using Tensor_Ptr = std::shared_ptr<Tensor>;

using TensorVec = std::vector<Tensor_Ptr>;

// sequence idx, kv_head idx
using kvCacheKey = std::tuple<int, int>;

}  // namespace llm_system