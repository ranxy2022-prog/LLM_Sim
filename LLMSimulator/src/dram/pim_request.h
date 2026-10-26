// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once

#include <list>
#include <memory>
#include <vector>

#include "base/type.h"
#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_object.h"

namespace llm_system {

/// @brief PIMRequest 的前向声明：此处只声明类型，完整结构见对应头文件。
class PIMRequest;

/// @brief 单条内存/PIM 命令描述：记录命令类型、操作数类别、分层地址和所属请求。
class PIMCommand {
 public:
  DRAMRequestType dramreq_type;
  PIMCommandType pimcmd_type;
  PIMOperandType op_type;
  Ramulator::AddrVec_t addr_vec;

  PIMRequest* request = NULL;

  /// @brief PIMCommand 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  PIMCommand() = default;

  /// @brief 保存命令类型、操作数角色和分层地址；带请求参数的重载同时关联所属 PIMRequest 及逻辑请求类型。
  PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
             Ramulator::AddrVec_t addr_vec);
  /// @brief 保存命令类型、操作数角色和分层地址；带请求参数的重载同时关联所属 PIMRequest 及逻辑请求类型。
  PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
             Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest);
  /// @brief 保存命令类型、操作数角色和分层地址；带请求参数的重载同时关联所属 PIMRequest 及逻辑请求类型。
  PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
             Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest,
             DRAMRequestType dramreq_type);
};

/// @brief 交给内存前端的一组 PIM/DRAM 命令，记录请求起止周期及执行命令计数。
class PIMRequest {
 public:
  friend class DRAMInterface;
  /// @brief 初始化请求的命令统计容器，供内存系统执行命令后回填计数与起止周期。
  PIMRequest();

  /// @brief 按枚举数量 resize 命令计数数组，新元素初始化为 0；已有同尺寸数组不会因此自动清零。
  void init();
  std::list<PIMCommand> command_queue;

  // stat
  std::vector<counter_t> issued_pim_cmd;
  std::vector<counter_t> issued_dram_cmd;

  cycle_t start = 0;
  cycle_t end = 0;

  /// @brief 把命令追加到队列，并增加对应 PIM 命令类型的生成次数。
  void AddCommand(PIMCommand&& pimcommand);
  /// @brief 返回命令队列的可修改引用，供内存前端消费命令。
  std::list<PIMCommand>& GetCommand();
};

}  // namespace llm_system
