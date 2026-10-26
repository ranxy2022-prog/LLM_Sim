// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#pragma once
#include <memory>
#include <vector>

#include "common/type.h"
#include "hardware/device.h"
#include "hardware/hardware_config.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief Cluster 的前向声明：此处只声明类型，完整结构见对应头文件。
class Cluster;

/// @brief 模拟节点：创建节点内的设备，并向这些设备转发执行、依赖设置与状态查询。
class Node : public std::enable_shared_from_this<Node> {
 public:
  using Ptr = std::shared_ptr<Node>;
  using Cluster_ptr = std::shared_ptr<Cluster>;

  /// @brief Node 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(SystemConfig config, int node_rank,
                                  Cluster_ptr cluster) {
    return Ptr(new Node(config, node_rank, cluster));
  }

  hw_metric node_ict_latency;
  hw_metric node_ict_bandwidth;
  int node_rank;

  std::vector<Device::Ptr> device;

  SystemConfig config;

  /// @brief 节点初始化的旧接口；当前项目中仅保留声明，未找到对应定义。
  void InitNode(int num_device);

  /// @brief Node 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  Node(Node &&) = default;
  /// @brief 使用编译器默认的成员赋值规则；智能指针复制共享所有权，移动成员按其类型规则转移状态。
  Node &operator=(Node &&) = default;

  /// @brief 依次让本节点所有设备建立执行图依赖。
  void set_dependency();
  /// @brief 只要节点中任一设备还有未执行模块，就返回 true。
  bool check_module_graph_remain();
  /// @brief 轮流推进本节点各设备的执行图；软件调用顺序不等于模拟设备必须串行运行。
  void run(std::vector<BatchedSequence::Ptr> sequences_metadata_list);

  /// @brief 向本节点设备传播性能估算开关。
  void setPerformExecution(bool perform);

  /// @brief 把全局设备 rank 转换为节点内索引，返回对应设备。
  Device::Ptr get_device(int device_total_rank);

 private:
  Cluster_ptr cluster;
  /// @brief 保存节点编号及节点内互连参数，并创建该节点的设备集合。
  Node(SystemConfig config, int node_rank, Cluster_ptr cluster);
  /// @brief Node 的默认/移动构造，按成员自身规则初始化；实际仿真通常使用带配置参数的构造路径。
  Node() = default;
  /// @brief 为节点内设备分配全局 rank，并创建关联同一 Cluster 的 Device 对象。
  void CreateDevice(SystemConfig config, Cluster_ptr cluster);
};

}  // namespace llm_system