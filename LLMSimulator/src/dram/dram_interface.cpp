// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include "dram/dram_interface.h"

#include <cmath>

#include "base/request.h"
#include "common/assert.h"
#include "dram/dram_request.h"

namespace llm_system {

/// @brief 加载 DRAM YAML，通过 Ramulator 工厂创建并连接前端/内存系统，注册命令生成内核和名称映射。
DRAMInterface::DRAMInterface(std::string config_path,
                             double memory_scale_factor)
    : memory_scale_factor(memory_scale_factor), exec_status() {
  std::vector<std::string> params;
  YAML::Node config = Ramulator::Config::parse_config_file(config_path, params);

  frontend = Ramulator::Factory::create_frontend(config);
  memory_system = Ramulator::Factory::create_memory_system(config);

  frontend->connect_memory_system(memory_system);
  memory_system->connect_frontend(frontend);

  frontend_tick = frontend->get_clock_ratio();
  mem_tick = memory_system->get_clock_ratio();

  tick_mult = frontend_tick * mem_tick;

  PIM_KERNEL::init(kernel);

  StringMapInit();
}

/// @brief 清零接口 ExecStatus，开始统计新一组内存请求。
void DRAMInterface::resetCounter() { exec_status = ExecStatus(); }

// Get requests and returns end time of each DRAM requests
/// @brief 逐个展开逻辑请求、提交命令并推进内存仿真，累计结果；start_time_cycle 参数当前未用于调度起始时刻。
void DRAMInterface::HandleRequest(const std::list<DRAMRequest::Ptr>& requests,
                                  cycle_t start_time_cycle) {
  resetCounter();
  for (auto& dram_req : requests) {
    PIMRequest pimrequest;

    // DRAM cycle to Core cycle
    GeneratePIMCommand(dram_req, pimrequest);
    SendRequest(pimrequest);
    run();
    updateStatus(pimrequest);
  }
}

/// @brief 把请求起止周期乘 memory_scale_factor 转为 ns，并按 HBM 命令索引累加 ACT/读写/刷新计数。
void DRAMInterface::updateStatus(const PIMRequest& pimrequest) {
  cycle_t duration = pimrequest.end - pimrequest.start;  // dram cycle

  time += (duration * memory_scale_factor);
  exec_status.memory_duration += (duration * memory_scale_factor);

  // Only for HBM, if you want to use another memory, (e.g. LPDDR5) order should be changed
  exec_status.act_count += pimrequest.issued_dram_cmd[0];
  exec_status.read_count += pimrequest.issued_dram_cmd[4];
  exec_status.write_count += pimrequest.issued_dram_cmd[5];
  exec_status.all_act_count += pimrequest.issued_dram_cmd[6];
  exec_status.all_read_count += pimrequest.issued_dram_cmd[7];
  exec_status.all_write_count += pimrequest.issued_dram_cmd[8];
  exec_status.ref_count += pimrequest.issued_dram_cmd[9];

  // For LPDDR5, order should be below
  /*
  exec_status.act_count += pimrequest.issued_dram_cmd[0];
  exec_status.read_count += pimrequest.issued_dram_cmd[9];
  exec_status.write_count += pimrequest.issued_dram_cmd[10];
  exec_status.all_act_count += pimrequest.issued_dram_cmd[2];
  exec_status.all_read_count += pimrequest.issued_dram_cmd[11];
  exec_status.all_write_count += pimrequest.issued_dram_cmd[12];
  exec_status.ref_count += pimrequest.issued_dram_cmd[15];
  */
}

/// @brief 把已展开的 PIMRequest 提交给 Ramulator 前端。
void DRAMInterface::SendRequest(PIMRequest& pimrequest) {
  frontend->send(pimrequest);
}

/// @brief 按逻辑请求类型选择已注册内核，将操作数展开为 PIMRequest 命令；空内核调用会打印未实现警告。
PIMRequest& DRAMInterface::GeneratePIMCommand(const DRAMRequest::Ptr request,
                                              PIMRequest& pimrequest) const {
  DRAMRequestType type = request->GetType();

  try {
    kernel[int(type)](pimrequest, type, request->operands_, pim_hw_config);
  } catch (std::bad_function_call) {
    notYetImplemented("PIM_KERNEL " + std::to_string(int(type)));
  }

  return pimrequest;
}

/// @brief 不断调用 memory_system->tick()，直到内存系统报告所有当前请求已完成。
void DRAMInterface::run() {
  for (uint64_t i = 0;; i++) {
    memory_system->tick();
    if (memory_system->is_finished()) {
      break;
    }
  }
}

/// @brief 建立请求、PIM 命令和操作数枚举到可读名称的映射，供调试输出。
void DRAMInterface::StringMapInit() {
  dramreq_to_string[DRAMRequestType::kRead] = "Read";
  dramreq_to_string[DRAMRequestType::kWrite] = "Write";
  dramreq_to_string[DRAMRequestType::kMove] = "Move";
  dramreq_to_string[DRAMRequestType::kMult] = "Mult";
  dramreq_to_string[DRAMRequestType::kAdd] = "Add";
  dramreq_to_string[DRAMRequestType::kMAD] = "MAD";
  dramreq_to_string[DRAMRequestType::kPMult] = "PMult";
  dramreq_to_string[DRAMRequestType::kCMult] = "CMult";
  dramreq_to_string[DRAMRequestType::kCAdd] = "CAdd";
  dramreq_to_string[DRAMRequestType::kCMAD] = "CMAD";
  dramreq_to_string[DRAMRequestType::kTensor] = "Tensor";
  dramreq_to_string[DRAMRequestType::kTensor_Square] = "Tensor_Square";
  dramreq_to_string[DRAMRequestType::kModup_Evkmult] = "Modup_Evkmult";
  dramreq_to_string[DRAMRequestType::kModDownEpilogue] = "ModDownEpilogue";
  dramreq_to_string[DRAMRequestType::kPMult_Accum] = "PMult_Accum";
  dramreq_to_string[DRAMRequestType::kCMult_Accum] = "CMult_Accum";

  pimcmd_to_string[PIMCommandType::kAdd] = "Add";
  pimcmd_to_string[PIMCommandType::kSub] = "Sub";
  pimcmd_to_string[PIMCommandType::kMult] = "Mult";
  pimcmd_to_string[PIMCommandType::kMAC] = "MAC";
  pimcmd_to_string[PIMCommandType::kDRAM2RF] = "DRAM2RF";
  pimcmd_to_string[PIMCommandType::kRF2DRAM] = "RF2DRAM";
  pimcmd_to_string[PIMCommandType::kRead] = "Read";
  pimcmd_to_string[PIMCommandType::kWrite] = "Write";

  pimoperand_to_string[PIMOperandType::kEvk] = "Evk";
  pimoperand_to_string[PIMOperandType::kModUp] = "ModUp";
  pimoperand_to_string[PIMOperandType::kRF] = "RF";
  pimoperand_to_string[PIMOperandType::kDRAM] = "DRAM";
}
}  // namespace llm_system