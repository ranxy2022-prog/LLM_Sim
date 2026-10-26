// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "hardware/device.h"
#include "module/base.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief ModuleGraph 的前向声明：此处只声明类型，完整结构见对应头文件。
class ModuleGraph;

/// @brief 计算模块基类：管理张量和子模块，以 forward 描述形状/成本传播，通过 operator() 记录执行图。
class Module : public std::enable_shared_from_this<Module> {
  friend class ModuleGraph;

 public:
  using Ptr = std::shared_ptr<Module>;

  /// @brief 保存模块名称、所属设备、参与设备列表及执行/同步标志，并生成用于跨设备查找的完整模块名。
  Module(std::string prefix = "", std::string name = "",
         Device::Ptr device = nullptr, std::vector<int> device_list = {},
         bool graph_execution = false, bool sync = false)
      : name(name),
        device(device),
        device_list(device_list),
        graph_execution(graph_execution),
        sync(sync) {
    module_map_name = prefix + "::" + name;
    if (sync) {
      sync_master = device_list.at(0);
    }
    need_to_aggregate = false;
  }

  /// @brief 虚析构入口，允许通过 Module 基类正确析构派生模块。
  virtual ~Module(){};

  /// @brief 通过 shared_from_this 返回当前 Module 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Module::Ptr getptr() { return shared_from_this(); }

  /// @brief 记录模块进入/退出执行图，并调用虚函数 forward；模型构图时用它建立层级与张量依赖。
  Tensor::Ptr operator()(const Tensor::Ptr input,
                         BatchedSequence::Ptr sequences_metadata);

  /// @brief 记录模块进入/退出执行图，并调用虚函数 forward；模型构图时用它建立层级与张量依赖。
  TensorVec operator()(const TensorVec input,
                       BatchedSequence::Ptr sequences_metadata);

  /// @brief 递归更新当前模块所持张量和子模块的设备关联。
  void set_device(Device::Ptr device);

  /// @brief 按指定 weight/act/cache 标签汇总模块及子模块的张量字节数，用于容量估算。
  long size(std::vector<std::string> tag = {"weight", "act", "cache"});

  /// @brief 查询该模块是否应在执行图回放时调用 forward；组合层与叶子算子的取值可能不同。
  bool execution() { return graph_execution; }
  /// @brief 把张量按名称加入本模块的 tensor_list，供 forward 和容量统计使用。
  void add_tensor(Tensor::Ptr tensor);

  /// @brief 把张量按名称加入本模块的 tensor_list，供 forward 和容量统计使用。
  void add_tensor(std::string name, Tensor::Ptr tensor);

  /// @brief 按名称取得子模块；使用 map::at，名称不存在时会抛出异常。
  Module::Ptr get_module(std::string module_name) {
    return module_list.at(module_name);
  }
  /// @brief 查找已有激活张量，按需要更新其形状/内存大小并设置就绪标记；不计算数值内容。
  Tensor::Ptr get_activation(std::string name, std::vector<int> shape = {1, 1},
                             bool ready = true);

  /// @brief 按缓存名、序列/头编号和压缩模式查找缓存张量，并重建其内存对象；当前 shape 参数未使用，不在此更新缓存形状。
  Tensor::Ptr get_cache(std::string name, int seq_idx, int kv_idx, bool compressed_kv,
                        std::vector<int> shape = {1, 1});

  /// @brief 把具体子类智能指针转换为 Module 指针，再登记为子模块。
  template <class T>
  void add_module(std::shared_ptr<T> module) {
    Module::Ptr _module = std::static_pointer_cast<Module>(module);
    add_module_with_name(module->name, _module);
  }

  /// @brief 通过集群查找参与设备的对应张量，填充本模块相关的依赖列表。
  void set_dependency_tensor(std::vector<Tensor::Ptr>& dependency_tensor_list,
                             Tensor::Ptr tensor);

  /// @brief 清除模块张量的就绪状态及相关本轮标记，供执行图重启使用。
  void unset_tensor();

  /// @brief 保存命名子模块并向设备/集群登记模块，供图执行和跨设备依赖查找。
  void add_module_with_name(std::string name, Module::Ptr module);

  /// @brief 设置模块的完整映射名称；该名称用于在各设备上定位同一逻辑模块。
  void set_module_map_name(std::string name) { module_map_name = name; }

  /// @brief 把当前模块的共享指针写入它持有的张量，建立张量到所属模块的反向关联。
  void set_tensor_module();

  std::string name;
  std::string module_map_name;
  bool sync;
  bool need_to_aggregate;

  /// @brief 递归汇总当前模块与子模块的权重、激活和缓存大小，返回分类字节数。
  std::vector<long> get_size();

 protected:
  std::vector<int> device_list;

  Device::Ptr device;
  int sync_master;
  std::map<std::string, Tensor::Ptr> tensor_list;
  // 模块树描述模型组成；执行图另外记录调用顺序，两者不是同一个容器。
  std::map<std::string, Module::Ptr> module_list;

  const bool graph_execution;
  // false 的组合模块在构图时展开子模块，正式回放通常由可执行的子模块承担实际成本。

 private:
  /// @brief 基类默认实现直接返回输入；具体算子重写此函数以更新形状、调用子模块或计算执行成本。
  virtual Tensor::Ptr forward(const Tensor::Ptr input,
                              BatchedSequence::Ptr sequences_metadata) {
    return input;
  }
  /// @brief 基类默认实现直接返回输入；具体算子重写此函数以更新形状、调用子模块或计算执行成本。
  virtual TensorVec forward(const TensorVec input,
                            BatchedSequence::Ptr sequences_metadata) {
    return input;
  }
};

}  // namespace llm_system
