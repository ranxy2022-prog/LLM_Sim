// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#pragma once

#include <list>
#include <memory>

#include "base/base.h"
#include "base/config.h"
#include "common/type.h"
#include "dram/dram_request.h"
#include "dram/dram_type.h"
#include "dram/pim_request.h"
#include "dram/pimkernel/pim_kernel.h"
#include "frontend/frontend.h"
#include "memory_system/memory_system.h"
#include "module/status.h"
#include "power.h"

namespace llm_system {

using pim_kernel_ptr = PIM_KERNEL::pim_kernel_ptr;

/// @brief 算子模型与 Ramulator 的适配层：把逻辑请求展开为命令，推进内存仿真并汇总周期及命令计数。
class DRAMInterface {
 public:
  friend class DataObject;

  using Ptr = std::shared_ptr<DRAMInterface>;

  /// @brief DRAMInterface 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] DRAMInterface::Ptr static Create(
      std::string config_path, const double memory_scale_factor) {
    return DRAMInterface::Ptr(
        new DRAMInterface(config_path, memory_scale_factor));
  }

  /// @brief DRAMInterface 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  DRAMInterface(DRAMInterface &&) = default;
  /// @brief 使用编译器默认的成员赋值规则；智能指针复制共享所有权，移动成员按其类型规则转移状态。
  DRAMInterface &operator=(DRAMInterface &&) = default;

  /// @brief 逐个展开逻辑请求、提交命令并推进内存仿真，累计结果；start_time_cycle 参数当前未用于调度起始时刻。
  void HandleRequest(const std::list<DRAMRequest::Ptr> &requests,
                     cycle_t start_time_cycle);

  /// @brief 不断调用 memory_system->tick()，直到内存系统报告所有当前请求已完成。
  void run();
  /// @brief 返回接口执行统计的可修改引用，调用方可读取、填充或覆盖它。
  ExecStatus &getExecStatus() { return exec_status; }
  /// @brief 按逻辑请求类型选择已注册内核，将操作数展开为 PIMRequest 命令；空内核调用会打印未实现警告。
  PIMRequest &GeneratePIMCommand(const DRAMRequest::Ptr request,
                                 PIMRequest &pimrequest) const;
  /// @brief 清零接口 ExecStatus，开始统计新一组内存请求。
  void resetCounter();

  PIMHWConfig pim_hw_config;

  /// @brief 设置后续命令展开使用的处理器类型和内部带宽倍率。
  void setPIMHWConfig(ProcessorType type, int bandwidth_x) {
    pim_hw_config.type = type;
    pim_hw_config.bandwidth_x = bandwidth_x;
  }

  std::map<DRAMRequestType, std::string> dramreq_to_string = {};
  std::map<PIMCommandType, std::string> pimcmd_to_string = {};
  std::map<PIMOperandType, std::string> pimoperand_to_string = {};

  time_ns time;

 private:
  Ramulator::IFrontEnd *frontend;
  Ramulator::IMemorySystem *memory_system;

  ExecStatus exec_status;

  long frontend_tick;
  long mem_tick;
  long tick_mult;
  double memory_scale_factor;

  std::vector<std::function<pim_kernel_ptr>> kernel;

  /// @brief 加载 DRAM YAML，通过 Ramulator 工厂创建并连接前端/内存系统，注册命令生成内核和名称映射。
  DRAMInterface(std::string config_path,
                double memory_scale_factor);

  /// @brief 把请求起止周期乘 memory_scale_factor 转为 ns，并按 HBM 命令索引累加 ACT/读写/刷新计数。
  void updateStatus(const PIMRequest &pimrequest);
  /// @brief 把已展开的 PIMRequest 提交给 Ramulator 前端。
  void SendRequest(PIMRequest &pimrequest);

  /// @brief 建立请求、PIM 命令和操作数枚举到可读名称的映射，供调试输出。
  void StringMapInit();

  std::vector<DRAMRequest> dram_request_log_;
  std::vector<DRAMRequest> pim_request_log_;
};

}  // namespace llm_system