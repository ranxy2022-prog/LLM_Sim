// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#pragma once
#include <map>
#include <memory>

#include "common/type.h"
#include "dram/dram_type.h"
#include "dram/memory_config.h"
#include "hardware/executor.h"
#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "module/status.h"
#include "scheduler/sequence.h"
#include "dram/power.h"

namespace llm_system {

/// @brief 模拟设备：持有硬件参数、执行图、地址映射和 DRAM 接口，记录设备级时间与执行状态。
class Device : public std::enable_shared_from_this<Device> {
  friend class Executor;
  friend class Cluster;

 public:
  using Ptr = std::shared_ptr<Device>;

  /// @brief 构造共享设备后把设备指针连接到执行图；连接操作需要对象已经受 shared_ptr 管理。
  [[nodiscard]] static Ptr Create(SystemConfig config, int device_total_rank,
                                  Cluster_ptr cluster) {
    Device::Ptr ptr = Ptr(new Device(config, device_total_rank, cluster));
    ptr->connectTopModuleGraph();
    return ptr;
  }

  hw_metric compute_peak_flops;
  hw_metric memory_bandwidth;
  hw_metric memory_capacity;

  SystemConfig config;

  // rank in node
  int device_local_rank;
  // rank in cluster
  int device_total_rank;

  /// @brief Device 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  Device() = default;

  /// @brief 通过 shared_from_this 返回当前 Device 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Device::Ptr get_ptr() { return shared_from_this(); }

  TopModuleGraph_ptr top_module_graph;

  /// @brief 把自身共享指针交给执行图，使执行图可以读取算子结果并更新设备状态。
  void connectTopModuleGraph();
  /// @brief 用默认 StatusBoard 重置设备的本轮累计时间、能耗等状态。
  void reset_status();
  /// @brief 清空模块级时间戳统计树，为新一轮记录做准备。
  void reset_timeboard();
  /// @brief 把依赖建立工作委托给本设备执行图。
  void set_dependency();

  /// @brief 查询本设备执行图是否还有未处理记录。
  bool check_module_graph_remain();
  /// @brief 把模块及其名称转交 Cluster，登记到本设备的模块索引表。
  void add_module(std::string name, Module_ptr module);

  /// @brief 复制模型配置到设备，供算子实现读取维度、精度和并行参数。
  void setModelConfig(ModelConfig& _model_config) {
    model_config = _model_config;
  }

  /// @brief 返回设备当前累计模拟时间，单位 ns。
  time_ns get_time() { return status.device_time; }
  /// @brief 覆盖设备累计模拟时间，单位 ns；用于时间线协调。
  void set_time(time_ns time) { status.device_time = time; }

  // run with module_graph
  /// @brief 按全局设备 rank 和非专家 TP 数找到 DP batch，使用该 batch 推进本设备执行图。
  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);

  /// @brief 把本设备执行图回到起点，重置张量就绪/时间戳状态以便下一轮执行。
  void restartGraph();

  /// @brief 设置是否启用硬件性能估算；构图阶段为 false，正式 run 阶段为 true。
  void setPerformExecution(bool perform) { perform_execution = perform; }

  // execute operations and update time;
  /// @brief 旧的三张量执行接口；当前仅有声明，正式路径使用带 LayerType、batch 元数据和 LayerInfo 的重载。
  void execution(Tensor_Ptr input, Tensor_Ptr weight, Tensor_Ptr output);
  /// @brief 旧的三张量执行接口；当前仅有声明，正式路径使用带 LayerType、batch 元数据和 LayerInfo 的重载。
  void execution(LayerType layer_type,
                 const std::vector<Tensor_Ptr>& tensor_list,
                 const BatchedSequence::Ptr sequences_metadata,
                 const LayerInfo layer_info);

  // allocate DataObject;
  /// @brief 把张量交给 MMapController，建立模拟内存地址对象；不分配实际 GPU 显存。
  void setMemoryObject(Tensor_Ptr tensor);

  /// @brief 把一个逻辑 DRAM 请求包装成列表，交给 DRAMInterface 执行。
  void run_ramulator(DRAMRequest_Ptr dram_request);
  /// @brief 按张量字节数、访问粒度、cube/伪通道数量及每行列数估算 ACT/读写次数；不遍历访问地址，也不模拟行冲突时序。
  void run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor);

  /// @brief 把性能结果按缓存键写入集群级缓存；键由算子、处理器、请求类型和数据大小等组成。
  void addExecutionCache(ExecStatus& exec_status, CacheKey key);

  /// @brief 把性能结果按缓存键写入集群级缓存；键由算子、处理器、请求类型和数据大小等组成。
  void addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                         ProcessorType processor_type,
                         DRAMRequestType dram_reqeust_type, long size);

  /// @brief 查询执行结果缓存，命中时复制到输出状态或 DRAM 接口状态；缓存不描述完整地址/行缓冲历史。
  bool checkExecutionCache(ExecStatus& exec_status, CacheKey key);
  /// @brief 查询执行结果缓存，命中时复制到输出状态或 DRAM 接口状态；缓存不描述完整地址/行缓冲历史。
  bool checkExecutionCache(CacheKey key);


  /// @brief 保存当前算子的执行结果，供执行图在模块退出时累计。
  void setExecStatus(ExecStatus& exec_status_);

  /// @brief 取走当前算子的执行结果，并将该暂存状态清零，避免重复累计。
  ExecStatus getExecStatus();
  /// @brief 取走高算力单元的暂存结果并清零该结果槽。
  ExecStatus getHighExecStatus();
  /// @brief 取走低算力单元的暂存结果并清零该结果槽。
  ExecStatus getLowExecStatus();

  /// @brief 向本设备执行图登记处理器对应的单次命令/运算能耗参数。
  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);

  Cluster_ptr cluster;
  DRAMInterface_Ptr dram_interface;

  StatusBoard status;
  ModelConfig model_config;

  MMapController_Ptr mmap_controller;

  bool perform_execution;

 private:
  ExecStatus high_exec_status;
  ExecStatus low_exec_status;
  ExecStatus exec_status;

  bool use_ramulator;

  /// @brief 保存设备配置/rank，创建执行图、地址映射器和 Ramulator 接口；初始关闭正式性能估算。
  Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster);

  /// @brief 旧调试入口：当前实现只取得输入/权重/输出并打印权重名称，未在这里发起完整仿真。
  void execution_ramulator(LayerType layer_type,
                           std::vector<Tensor_Ptr> tensor_list);

  /// @brief 旧的理想执行接口；当前项目中仅保留声明，未找到对应定义。
  void execution_ideal(LayerType layer_type,
                       std::vector<Tensor_Ptr> tensor_list);
};
}  // namespace llm_system