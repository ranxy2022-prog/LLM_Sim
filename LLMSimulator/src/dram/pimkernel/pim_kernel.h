// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once

#include <functional>
#include <list>
#include <memory>
#include <vector>

#include "base/type.h"
#include "common/assert.h"
#include "dram/dram_request.h"
#include "dram/pim_request.h"
#include "hardware/hardware_config.h"

namespace llm_system {
namespace PIM_KERNEL {

template <typename... Args>
std::vector<std::function<Args...>> _PIM_KERNEL;
// auto kernel = _PIM_KERNEL<void(DRAMRequest, PIMRequest&, const
// pinballConfig)>;

using pim_kernel_ptr = void(PIMRequest&, DRAMRequestType,
                            DRAMRequest::PIM_Operand&, PIMHWConfig);

// using pim_kernel_ptr = DRAMInterface::pim_kernel_ptr;
//    void(PIMRequest&, DRAMRequestType,
//                              DRAMRequest::PIM_Operand&, pinballConfig);
//  //auto kernel = _PIM_KERNEL<pim_kernel_ptr>;

// auto kernel = _PIM_KERNEL<void(PIMRequest&, DRAMRequestType,
//   DRAMRequest::PIM_Operand&, pinballConfig)>;

// std::vector<std::function<void(DRAMRequest, PIMRequest&)>> PIM_KERNEL;

/// @brief 从逻辑请求的操作数表中取出指定角色的 MemoryObject 列表。
std::vector<MemoryObject::Ptr> get_operand(
    const DRAMRequest::PIM_Operand& operand, const PIMOperandType type);

/// @brief 注册逻辑内存请求的命令生成函数；当前注册范围以函数内赋值为准，并非全部枚举均已实现。
void init(std::vector<std::function<pim_kernel_ptr>>& kernel);

/// @brief 要求一个 DRAM 操作数，按访问块生成读命令；根据处理器选择地址映射，仅提交地址向量首项为 0 的访问，非 GEMM tiling 轨迹。
pim_kernel_ptr Read_kernel;
/// @brief 遍历操作数的访问块并生成写命令，按实现的地址筛选与处理器分支构造请求。
pim_kernel_ptr Write_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr Move_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr Mult_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr Add_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr MAD_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr PMult_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr CMult_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr CAdd_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr CMAD_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr Tensor_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr Tensor_Square_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr ModUp_Evkmult_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr ModDownEpilogue_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr PMult_Accum_kernel;
/// @brief 预留的 PIM 命令生成函数声明；当前自有源码未提供此内核的定义，枚举/声明不代表可用实现。
pim_kernel_ptr CMult_Accum_kernel;
/// @brief 根据处理器类型和带宽倍率把权重访问展开为读/计算相关命令，模拟 GEMV 访存路径而非数值乘加。
pim_kernel_ptr GEMV;

}  // namespace PIM_KERNEL

}  // namespace llm_system
