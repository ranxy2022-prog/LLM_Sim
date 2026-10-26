// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#include "hardware/node.h"

namespace llm_system {

/// @brief 保存节点编号及节点内互连参数，并创建该节点的设备集合。
Node::Node(SystemConfig config, int node_rank, Cluster_ptr cluster)
    : config(config), node_rank(node_rank), cluster(cluster) {
  node_ict_latency = config.device_ict_latency;
  node_ict_bandwidth = config.device_ict_bandwidth;

  CreateDevice(config, cluster);
}

/// @brief 把全局设备 rank 转换为节点内索引，返回对应设备。
Device::Ptr Node::get_device(int device_total_rank) {
  int local_rank = device_total_rank % config.num_device;
  return device.at(local_rank);
}

/// @brief 依次让本节点所有设备建立执行图依赖。
void Node::set_dependency() {
  for (Device::Ptr _device : device) {
    _device->set_dependency();
  }
}

/// @brief 只要节点中任一设备还有未执行模块，就返回 true。
bool Node::check_module_graph_remain() {
  for (Device::Ptr _device : device) {
    if (_device->check_module_graph_remain()) {
      return true;
    }
  }
  return false;
}

/// @brief 轮流推进本节点各设备的执行图；软件调用顺序不等于模拟设备必须串行运行。
void Node::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  for (Device::Ptr _device : device) {
      _device->run(sequences_metadata_list);
  }
}

/// @brief 向本节点设备传播性能估算开关。
void Node::setPerformExecution(bool perform) {
  for (Device::Ptr _device : device) {
    _device->setPerformExecution(perform);
  }
};

/// @brief 为节点内设备分配全局 rank，并创建关联同一 Cluster 的 Device 对象。
void Node::CreateDevice(SystemConfig config, Cluster_ptr cluster) {
  int device_rank_offset = node_rank * config.num_device;
  for (int device_rank = 0; device_rank < config.num_device; device_rank++) {
    device.push_back(
        Device::Create(config, device_rank + device_rank_offset, cluster));
  }
}

};  // namespace llm_system