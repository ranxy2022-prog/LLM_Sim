// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once
#include "common/type.h"

namespace llm_system {

/// @brief 张量的模拟内存分布方式，用于选择地址分配与映射路径。
enum class MMap {
  ALL_CHANNEL = 0,
  CUBE_CHANNEL,
  CHANNEL,
};

/// @brief 逻辑内存/PIM 请求种类；枚举存在不代表对应命令生成内核已经实现。
enum class DRAMRequestType {
  kRead = 0,   // None-pim
  kWrite = 1,  // None-pim
  kMove = 2,
  kMult = 3,
  kAdd = 4,
  kMAD = 5,
  kPMult = 6,
  kCMult = 7,
  kCAdd = 8,
  kCMAD = 9,
  kTensor = 10,
  kTensor_Square = 11,
  kModup_Evkmult = 12,
  kModDownEpilogue = 13,
  kPMult_Accum = 14,
  kCMult_Accum = 15,
  kGEMV = 16,
  kMAX,
};

/// @brief PIM 命令类型，供请求展开和内存前端识别读写、搬运及计算操作。
enum class PIMCommandType {
  kAdd = 0,  // r0 = r1 + r2
  kSub,      // r0 = r1 - r2
  kMult,     // r0 = r1 * r2
  kMAC,      // r0 = r1 * r2 + r3
  kDRAM2RF,  // Move, DRAM to register file
  kRF2DRAM,  // Move, register file to DRAM
  kRead,     // Generic read
  kWrite,    // Generic Write
  kMAX,
};

/// @brief HBM/DRAM 命令类别，对应 ACT、读写、刷新等计数与能耗项目。
enum class DRAMCommandType {
  kACT = 0,
  kPRE,
  kPREA,
  kALL_PRE,
  kREAD,
  kWRITE,
  kALL_ACT,
  kALL_READ,
  kALL_WRITE,
  kREF,
  kMAX,
};

/// @brief LPDDR5 命令类别；当前默认 HBM 统计不能直接复用其枚举索引。
enum class LPDDR5CommandType { // LPDDR5
  kACT_1 = 0,
  kACT_2,
  kALL_ACT_1,
  kALL_ACT_2,
  kPRE,
  kPREA,
  kALL_PRE,
  kCASRD,
  kCASWR,
  kRD16,
  kWR16,
  kRD16A,
  kWR16A,
  kALL_RD,
  kALL_WR,
  kREF,
  kMAX,
};

/// @brief 内存请求中的操作数角色，用于从请求的操作数表中选择源、目标或存储区域。
enum class PIMOperandType {
  kRF = 0,
  kDRAM = 1,
  kSrc = 2,
  kPrecomputed = 3,
  kDest = 4,
  kModUp = 5,  // deprecated
  kEvk = 6,    // deprecated
  kMAX = 7,
};

}  // namespace llm_system
