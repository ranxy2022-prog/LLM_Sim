// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include "dram/pimkernel/pim_kernel.h"

#include <stdexcept>
namespace llm_system {
namespace PIM_KERNEL {

/// @brief 从逻辑请求的操作数表中取出指定角色的 MemoryObject 列表。
std::vector<MemoryObject::Ptr> get_operand(
    const DRAMRequest::PIM_Operand& operand, const PIMOperandType type) {
  std::vector<MemoryObject::Ptr> operands;
  try {
    operands = operand.at(type);
  } catch (std::out_of_range) {
    fail("get_operand: invalid operand");
  }
  return operands;
}

/// @brief 注册逻辑内存请求的命令生成函数；当前注册范围以函数内赋值为准，并非全部枚举均已实现。
void init(std::vector<std::function<pim_kernel_ptr>>& kernel) {
  // PIM_KERNEL.resize(DRAMRequestType::kMAX, std::function<void(DRAM))

  kernel.resize((int)DRAMRequestType::kMAX);

  kernel[(int)DRAMRequestType::kRead] = &Read_kernel;
  kernel[(int)DRAMRequestType::kWrite] = &Write_kernel;
  //  kernel[(int)DRAMRequestType::kMove] = &Move_kernel;
  //  kernel[(int)DRAMRequestType::kMult] = &Mult_kernel;
  //  kernel[(int)DRAMRequestType::kAdd] = &Add_kernel;
  //  kernel[(int)DRAMRequestType::kMAD] = &MAD_kernel;
  //  kernel[(int)DRAMRequestType::kPMult] = &PMult_kernel;
  //  kernel[(int)DRAMRequestType::kCMult] = &CMult_kernel;
  //  kernel[(int)DRAMRequestType::kCAdd] = &CAdd_kernel;
  //  kernel[(int)DRAMRequestType::kCMAD] = &CMAD_kernel;
  //  kernel[(int)DRAMRequestType::kTensor] = &Tensor_kernel;
  //  kernel[(int)DRAMRequestType::kTensor_Square] = &Tensor_Square_kernel;
  //  kernel[(int)DRAMRequestType::kModup_Evkmult] = &ModUp_Evkmult_kernel;
  //  kernel[(int)DRAMRequestType::kModDownEpilogue] = &ModDownEpilogue_kernel;
  //  kernel[(int)DRAMRequestType::kPMult_Accum] = &PMult_Accum_kernel;
  //  kernel[(int)DRAMRequestType::kCMult_Accum] = &CMult_Accum_kernel;
  kernel[(int)DRAMRequestType::kGEMV] = &GEMV;
}

}  // namespace PIM_KERNEL

}  // namespace llm_system
