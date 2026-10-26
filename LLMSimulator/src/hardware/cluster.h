// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#pragma once
#include <memory>
#include <tuple>
#include <vector>

#include "common/type.h"
#include "dram/dram_type.h"
#include "hardware/hardware_config.h"
#include "hardware/node.h"
#include "hardware/stat.h"
#include "module/module.h"
#include "scheduler/scheduler.h"
#include "dram/power.h"

namespace llm_system {

/// @brief 集群级仿真控制器：组织节点/设备、共享算子执行器，驱动多轮仿真并导出统计。
class Cluster : public std::enable_shared_from_this<Cluster> {
  friend class Device;

 public:
  using Ptr = std::shared_ptr<Cluster>;
  /// @brief 创建共享集群后建立节点/设备，并分别登记 GPU、LOGIC、PIM 的命令能耗参数。
  [[nodiscard]] static Ptr Create(SystemConfig config,
                                  Scheduler::Ptr scheduler) {
    Ptr cluster = Ptr(new Cluster(config, scheduler));
    cluster->set(config);
    cluster->initializeDRAM((int)(ProcessorType::GPU), gpuEnergy);
    cluster->initializeDRAM((int)(ProcessorType::LOGIC), logicEnergy);
    cluster->initializeDRAM((int)(ProcessorType::PIM), pimEnergy);
    return cluster;
  }

  hw_metric cluster_ict_latency;
  hw_metric cluster_ict_bandwidth;

  std::vector<Node::Ptr> node;

  /// @brief 通过 shared_from_this 返回当前 Cluster 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Ptr getptr() { return shared_from_this(); }

  /// @brief Cluster 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  Cluster(Cluster &&) = default;
  /// @brief 使用编译器默认的成员赋值规则；智能指针复制共享所有权，移动成员按其类型规则转移状态。
  Cluster &operator=(Cluster &&) = default;

  SystemConfig config;
  int num_device;
  int num_node;
  int num_total_device;
  bool out_of_memory = false;

  /// @brief 把全局设备编号转换为节点编号，再从该节点取得设备对象。
  Device::Ptr get_device(int device_total_rank);

  /// @brief 让所有节点/设备建立执行图依赖，使通信模块可找到其他设备上的对应张量。
  void set_dependency();
  /// @brief 把设备上的模块按完整名称登记到 module_map，重复名称会触发错误。
  void add_module(int device_rank, std::string name, Module::Ptr module);

  /// @brief 根据模块完整名称和张量名称，收集参与设备上的对应激活张量，建立跨设备依赖。
  void set_dependency_tensor(std::vector<Tensor::Ptr> &list, Tensor::Ptr tensor,
                             const std::vector<int> &device_list);

  /// @brief 正式多轮仿真入口：创建 CSV、填充并预热请求队列，再按 disagg_system 选择普通或 PD 分离循环。
  std::vector<Stat> runIteration(int iter, std::string file_name = "stat");

  /// @brief 逐轮分配 token、执行模型、统计并更新请求；CSV 在轮首刷新，当前实现没有在循环结束时刷新最后一批记录。
  std::vector<Stat> runIterationMixed(int iter, std::ofstream &csv);
  /// @brief 用独立 prefill/decode 时间记账近似 PD 分离；仍复用同一集群执行模型，非两套独立硬件，CSV 每 25 轮阶段性刷新。
  std::vector<Stat> runIterationSumGenSplit(int iter, std::ofstream &csv);

  /// @brief 执行一轮：打开性能估算开关，重置图/统计，然后轮转推进各节点，直到所有设备的执行图完成。
  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);
  /// @brief 重置各设备执行图位置、累计状态和模块时间戳，为新一轮仿真做准备。
  void restartModuleGraph();

  /// @brief 把指定处理器的命令能耗系数传给各设备，用于将命令次数换算成能耗。
  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);

  /// @brief 向节点和设备传播是否真正调用性能模型的开关，用于区分构图与正式执行。
  void setPerformExecution(bool perform);

  /// @brief 时间戳更新接口；当前项目中仅保留声明，未找到对应函数定义。
  void updateTimestamp();
  /// @brief 内存池加载时间校正接口；当前项目中仅保留声明，未找到对应函数定义。
  void calibrateMemPoolLoadTime(Stat &stat);

  std::vector<std::map<std::string, Module::Ptr>> module_map;

  /// @brief 估计权重、激活和 KV cache 的设备容量需求；按配置报告 OOM 或调整 batch 容量并重建队列。
  bool checkMemorySize();
  /// @brief 检查异构执行相关的容量需求，并按配置调整可容纳的 batch；属于容量估算而非真实显存分配。
  bool checkHeteroMemorySize();
  /// @brief 汇总各设备的 DRAM 与计算能耗，返回供 Stat/CSV 填充的分类能耗数组。
  std::vector<energy_nJ> getTotalEnergy();

  /// @brief 在设备 0 的时间戳树中查找各类模型子模块，汇总投影、注意力、FFN/MoE、通信等耗时与能耗。
  void setTimeBreakDown(Stat &stat);
  /// @brief 从调度器和设备取得 batch、token、上下文及阶段耗时，填充当前统计记录。
  void setStat(Stat &stat);

  /// @brief 把首次 token/请求结束事件转换为请求级统计记录，同时更新请求的生成起始时刻。
  void addLatency(std::vector<Stat> &stat_list,
                  const std::vector<Sequence::Ptr> &seq_list, time_ns time);

  /// @brief 导出各设备的文本时间线；会创建目录并先删除目标目录中的全部现有条目，应使用专用输出目录。
  void exportGantt(std::string gantt_file_path);

  std::map<CacheKey, ExecStatus> execution_time_cache;

 private:
  /// @brief 检查是否还有设备的执行图未走完，控制单轮 Cluster::run 的推进循环。
  bool check_module_graph_remain();
  /// @brief 把暂存 Stat 记录写入已打开的 CSV 流，然后清空 stat_list；返回列表因此不一定包含所有轮次。
  void exportToCSV(std::ofstream &csv, std::vector<Stat> &stat_list);
  /// @brief 根据配置创建节点，并为每个设备分配模块名称索引表。
  void set(SystemConfig config);
  /// @brief 保存系统配置和共享调度器，记录节点/设备数及节点间互连参数；节点由后续 set 创建。
  Cluster(SystemConfig config, Scheduler::Ptr scheduler);
  /// @brief Cluster 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  Cluster() = default;
  Executor executor;

  Scheduler::Ptr scheduler;

  /// @brief 按 num_node 创建节点对象，把当前集群的共享指针传给节点以建立关联。
  void CreateNode(SystemConfig config);
};

}  // namespace llm_system