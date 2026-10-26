// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iomanip>
#include <vector>

#include "common/assert.h"
#include "common/type.h"
#include "iostream"
#include "module/status.h"

namespace llm_system {

/// @brief 模块级统计树节点：保存进入/退出状态之差，表达模块及其子模块的耗时、张量形状和能耗。
class TimeStamp {
 public:
  // TimeStamp() = default;
  /// @brief 记录模块进入时的统计板快照、名称、父节点和层级，用于退出时计算该模块的增量成本。
  TimeStamp(std::string name = "", const StatusBoard& status = StatusBoard(),
            TimeStamp* parent = nullptr, int level = -1);

  /// @brief 创建子时间戳并记录起始状态，返回子节点供后续嵌套模块使用。
  TimeStamp* push_stamp(std::string name, const StatusBoard& status) {
    TimeStamp* stamp = new TimeStamp(name, status, this, level + 1);
    leafstamp.push_back(stamp);
    return stamp;
  }

  /// @brief 用退出状态结算当前模块，然后返回父时间戳节点。
  TimeStamp* pop_stamp(const StatusBoard& output_status) {
    set_status(output_status);
    return parent;
  }
  /// @brief 递归收集名称匹配的时间戳节点，供按算子类别汇总成本。
  void find_stamp(std::string name, std::vector<TimeStamp*> &stamp_vec);

  /// @brief 用模块退出时的累计状态减去进入快照，得到该模块时间、能耗、运算量及输入输出信息。
  void set_status(const StatusBoard& output_status);

  /// @brief 返回该模块 end_time - start_time，单位 ns。
  time_ns get_duration(){
    return status.end_time - status.start_time;
  }

  /// @brief 求和普通及全 bank 的 ACT/读/写能耗，不包含计算 MAC 能耗。
  energy_nJ getDramEnergy(){
    return status.act_energy + status.read_energy + status.write_energy +
    status.all_act_energy + status.all_read_energy + status.all_write_energy;
  }

  /// @brief 求和标记为加载部分的 ACT/读/写能耗。
  energy_nJ getDramEnergyForLoad(){
    return status.act_energy_load + status.read_energy_load + status.write_energy_load +
    status.all_act_energy_load + status.all_read_energy_load + status.all_write_energy_load;
  }

  /// @brief 返回当前模块记录的 MAC/计算能耗。
  energy_nJ getCompEnergy(){
    return status.mac_energy;
  }

  /// @brief 返回记录的运算量/访存量比值，用于分析算术强度。
  double getOpb() { return status.opb; }

  /// @brief 打印当前模块的时间、形状、利用率、能耗，并递归输出子模块。
  void print();
  /// @brief 把模块时间区间按设备写入文本文件，并递归处理子时间戳。
  void exportGantt(std::string filepath, int device_id);

  /// @brief 按模块层级打印起止时间及持续时间，日志显示单位按实现转换。
  void print_time();
  /// @brief 打印模块输入与输出张量形状，张量列表按对应分支展示。
  void print_tensor();
  /// @brief 打印模块记录的计算/访存利用率与相关性能指标。
  void print_util();
  /// @brief 打印模块能耗分类；此处显示的 Total 是内存项之和，不应当作包含 MAC 的完整总能耗。
  void print_energy();

  /// @brief 释放并清空所有子时间戳，子节点析构会继续释放后代。
  void deleteTimeStamp() {
    for (auto& stamp : leafstamp) {
      delete stamp;
    }
    leafstamp.clear();
  }

  /// @brief 递归删除子时间戳，释放统计树节点。
  ~TimeStamp() {
    for (auto& stamp : leafstamp) {
      delete stamp;
    }
    leafstamp.clear();
  }

 private:
  int level = -1;
  std::string name;

  time_ns start_time;
  time_ns end_time;

  StatusBoard status;

  std::vector<TimeStamp*> leafstamp;
  TimeStamp* parent;
};

/// @brief 模块统计树的管理器：维护当前时间戳位置，支持进入/退出、重置、查找与输出。
class TimeBoard {
 public:
  /// @brief 建立根时间戳并把当前指针指向根；复制构造被禁用以避免原始节点指针重复管理。
  TimeBoard() : top() { current_stamp = &top; }

  /// @brief 禁止复制该对象，避免内部状态或原始指针被重复管理。
  TimeBoard(const TimeBoard& copy) = delete;

  /// @brief 进入一个模块：在当前节点下建立子时间戳，并移动当前记录位置。
  void push_timestamp(const StatusBoard& status, std::string name) {
    assertTrue(current_stamp, "No stamp in TimeBoard");
    current_stamp = current_stamp->push_stamp(name, status);
  };

  /// @brief 结束当前模块的统计，并退回父时间戳。
  void pop_timestamp(const StatusBoard& status) {
    current_stamp = current_stamp->pop_stamp(status);
  }

  /// @brief 删除旧的子时间戳树并回到根节点，供下一轮记录。
  void reset_timeboard() {
    top.deleteTimeStamp();
    // print();
    //  top = TimeStamp();
    //   status = StatusBoard();
    current_stamp = &top;
  }

  // void set_time(time_ns time) { status.time = time; }
  // void add_time(time_ns time) { status.time += time; }

  /// @brief 从根节点递归打印模块性能报告。
  void print() { top.print(); }

  /// @brief 从根节点递归导出指定设备的文本时间线。
  void exportGantt(std::string filepath, int device_id) {
    top.exportGantt(filepath, device_id);
  };

  /// @brief 从根节点按名称查找时间戳，结果追加到调用方提供的数组中。
  void find_stamp(std::string name, std::vector<TimeStamp*> &stamp_vec){
    top.find_stamp(name, stamp_vec);
  }

  // time_ns get_time() { return status.time; }

 private:
  TimeStamp top;
  // StatusBoard status;
  TimeStamp* current_stamp;
};

}  // namespace llm_system