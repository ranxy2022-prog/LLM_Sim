// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include "dram/pim_request.h"

#include "common/assert.h"

namespace llm_system {

/// @brief 保存命令类型、操作数角色和分层地址；带请求参数的重载同时关联所属 PIMRequest 及逻辑请求类型。
PIMCommand::PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
                       Ramulator::AddrVec_t addr_vec)
    : pimcmd_type(pimcmd_type), op_type(op_type), addr_vec(addr_vec) {}

/// @brief 保存命令类型、操作数角色和分层地址；带请求参数的重载同时关联所属 PIMRequest 及逻辑请求类型。
PIMCommand::PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
                       Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest)
    : pimcmd_type(pimcmd_type),
      op_type(op_type),
      addr_vec(addr_vec),
      request(pimrequest) {}

/// @brief 保存命令类型、操作数角色和分层地址；带请求参数的重载同时关联所属 PIMRequest 及逻辑请求类型。
PIMCommand::PIMCommand(PIMCommandType pimcmd_type, PIMOperandType op_type,
                       Ramulator::AddrVec_t addr_vec, PIMRequest* pimrequest,
                       DRAMRequestType dramreq_type)
    : pimcmd_type(pimcmd_type),
      op_type(op_type),
      addr_vec(addr_vec),
      dramreq_type(dramreq_type),
      request(pimrequest) {}

/// @brief 初始化请求的命令统计容器，供内存系统执行命令后回填计数与起止周期。
PIMRequest::PIMRequest() { init(); }

/// @brief 按枚举数量 resize 命令计数数组，新元素初始化为 0；已有同尺寸数组不会因此自动清零。
void PIMRequest::init() {
  issued_pim_cmd.resize((int)PIMCommandType::kMAX, 0);
  issued_dram_cmd.resize((int)DRAMCommandType::kMAX, 0);
}

/// @brief 把命令追加到队列，并增加对应 PIM 命令类型的生成次数。
void PIMRequest::AddCommand(PIMCommand&& pimcommand) {
  command_queue.push_back(pimcommand);

  // update the number of issued pimcmd
  issued_pim_cmd[(int)pimcommand.pimcmd_type]++;
}

/// @brief 返回命令队列的可修改引用，供内存前端消费命令。
std::list<PIMCommand>& PIMRequest::GetCommand() { return command_queue; }

}  // namespace llm_system