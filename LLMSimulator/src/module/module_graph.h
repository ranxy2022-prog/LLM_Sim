// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <map>
#include <string>

#include "common/assert.h"
#include "hardware/cluster.h"
#include "module/base.h"
#include "module/module.h"
#include "module/status.h"
#include "module/timeboard.h"
#include "scheduler/scheduler.h"
#include "dram/power.h"

namespace llm_system {

// block of modules which can run simultaneosuly without communication and
// synchronizations

/// @brief TopModuleGraph 的前向声明：此处只声明类型，完整结构见对应头文件。
class TopModuleGraph;

/// @brief 执行图中的一个模块记录（进入或退出）：保存输入与依赖，检查就绪状态并触发模块 forward。
class ModuleGraph : std::enable_shared_from_this<ModuleGraph> {
 public:
  friend class TopModuleGraph;
  using Ptr = std::shared_ptr<ModuleGraph>;

  /// @brief ModuleGraph 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(StatusBoard& status,
                                  Module::Ptr module = nullptr,
                                  Tensor::Ptr input = nullptr,
                                  int module_level = 0,
                                  bool module_pop = false) {
    return Ptr(
        new ModuleGraph(module, status, input, module_level, module_pop));
  };

  /// @brief ModuleGraph 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(StatusBoard& status,
                                  Module::Ptr module = nullptr,
                                  TensorVec input = {}, int module_level = 0,
                                  bool module_pop = false) {
    return Ptr(
        new ModuleGraph(module, status, input, module_level, module_pop));
  };

  /// @brief 非执行记录直接通过；可执行模块在依赖就绪后调用 forward，否则返回 false 让设备暂时停止推进。
  bool run(BatchedSequence::Ptr sequences_metadata);

  /// @brief 打印当前执行记录的模块名称、层级和形状等调试信息。
  void print_graph();

  /// @brief 返回关联模块的短名称。
  std::string getname() { return module->name; };

  /// @brief 同步模块等待跨设备依赖并对齐时间，普通模块检查输入就绪状态，决定能否执行。
  bool check_ready();
  /// @brief 把单张量输入标记为已就绪。
  void set_ready() { input->set(); };

  /// @brief 建立该图记录的张量依赖列表。
  void set_dependency();

  /// @brief 清除所属模块的张量就绪标记，并清除本记录已记时间戳标志。
  void unset_tensor() {
    if (module) {
      module->unset_tensor();
    }
    unset_stamped();
  }

  /// @brief 查询本轮是否已记录该节点的进入/退出时间戳，避免阻塞重试时重复记账。
  bool is_stamped() { return stamped; }
  /// @brief 清除本轮时间戳完成标志。
  void unset_stamped() { stamped = false; }
  /// @brief 标记本轮已经为该图记录采集时间戳。
  void set_stamped() { stamped = true; }

  /// @brief 返回关联模块的短名称，供时间戳树记录。
  std::string get_name() { return module->name; }

  /// @brief 为需要跨设备同步的模块取得对应依赖张量，供 ready 检查和时间同步。
  void set_dependency_tensor();

  /// @brief 查询当前记录是否为模块退出记录；退出时需要汇总成本并结束时间戳。
  bool is_pop() { return module_pop; }

  /// @brief 根据同步依赖将相关设备时间线对齐，避免消费者早于生产者完成时刻执行。
  void sync_devices();

  /// @brief 检查 dependency_tensor_list 中所有张量是否就绪；当前实现没有使用传入的 tensor_list 参数。
  bool checkListReady(TensorVec tensor_list);

  std::vector<Tensor::Ptr> dependency_tensor_list;

 private:
  /// @brief 保存模块、输入、统计板引用及层级/退出标志，初始化本轮时间戳标记。
  ModuleGraph(Module::Ptr module, StatusBoard& status, Tensor::Ptr input,
              int module_level, bool module_pop);
  /// @brief 保存模块、输入、统计板引用及层级/退出标志，初始化本轮时间戳标记。
  ModuleGraph(Module::Ptr module, StatusBoard& status, TensorVec input,
              int module_level, bool module_pop);

  Tensor::Ptr input;
  TensorVec input_vec;
  Tensor::Ptr output;
  TensorVec output_vec;

  bool isTensorVec;

  int module_level;

  bool module_pop;
  bool stamped;

  StatusBoard& status;

  Module::Ptr module;
};

/// @brief 一台设备的执行记录序列：顺序推进模块，遇依赖阻塞暂停，管理同步、时间戳与累计开销。
class TopModuleGraph {
 public:
  using Ptr = std::shared_ptr<TopModuleGraph>;

  /// @brief TopModuleGraph 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(StatusBoard& status) {
    return Ptr(new TopModuleGraph(status));
  };

  std::vector<ModuleGraph::Ptr> module_graph;

  /// @brief 保存所属设备的共享指针，以便取得执行结果并协调时间线。
  void connectDevice(Device::Ptr device_) { device = device_; }

  /// @brief 在构图时追加模块进入记录，保存单张量或张量列表输入，并增加层级。
  void push_module_graph(Module::Ptr module, Tensor::Ptr input);
  /// @brief 在构图时追加模块退出记录，保存输出描述并恢复父层级。
  void pop_module_graph(Tensor::Ptr input);

  /// @brief 在构图时追加模块进入记录，保存单张量或张量列表输入，并增加层级。
  void push_module_graph(Module::Ptr module, TensorVec input);
  /// @brief 在构图时追加模块退出记录，保存输出描述并恢复父层级。
  void pop_module_graph(TensorVec input);

  /// @brief 从当前迭代器顺序执行记录并采集时间戳；遇到未满足的依赖即暂停，供集群稍后重试。
  void run(BatchedSequence::Ptr sequences_metadata);
  /// @brief 回到执行序列起点，清除各记录时间戳和张量就绪状态，为下一轮回放做准备。
  void restart_graph();

  /// @brief 遍历执行记录，建立需要的模块/跨设备张量依赖。
  void set_dependency();
  /// @brief 遍历并打印设备执行图记录，帮助检查模型构建顺序。
  void print_graph();

  /// @brief 清空本设备模块时间戳树。
  void reset_timeboard() {
    timeboard.reset_timeboard();
  }

  /// @brief 登记某一处理器类型的命令能耗系数。
  void initializeDRAM(int ProcessorType, DramEnergy dramEnergy);
  /// @brief 汇总设备 DRAM 各类命令能耗与计算能耗，返回分类数组。
  std::vector<energy_nJ> getDeviceEnergy();

  /// @brief 比较当前执行位置与序列末尾，判断是否还有未完成记录。
  bool check_module_graph_remain() {
    if (current_module == module_graph.end()) {
      return false;
    } else {
      return true;
    }
  }

  /// @brief 输出本设备模块层级的耗时、形状及能耗报告。
  void print_timeboard() { timeboard.print(); }

  /// @brief 把本设备时间戳树交给 TimeBoard，导出文本时间线。
  void exportGantt(std::string filepath, int device_id) {
    timeboard.exportGantt(filepath, device_id);
  };

  Device::Ptr device;

  TimeBoard timeboard;

 private:
  /// @brief 关联设备统计板并初始化执行记录容器和时间戳管理状态。
  TopModuleGraph(StatusBoard& status);
  /// @brief 进入模块前更新统计板的输入形状与异构时间线状态，作为时间戳起始快照。
  void set_push_status();
  /// @brief 取走算子 ExecStatus，累计 device/high/low 时间及命令/计算能耗，更新输出形状统计。
  void set_pop_status();

  /// @brief 根据进入/退出记录采集时间戳；用 stamped 标志避免依赖阻塞重试时重复累计。
  void set_stamp();
  /// @brief 已废弃接口：当前调用会触发 fail，正常路径使用 set_stamp。
  void push_stamp();
  /// @brief 已废弃接口：当前调用会触发 fail，正常路径使用 set_stamp。
  void pop_stamp();

  StatusBoard& status;

  std::vector<DramEnergy> dram_powers;

  int current_module_level = 0;
  std::vector<ModuleGraph::Ptr>::iterator current_module;
};

}  // namespace llm_system