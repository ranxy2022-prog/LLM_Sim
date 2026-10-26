// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/module.h"

#include <iostream>

#include "common/assert.h"
#include "module/module_graph.h"
#include "module/tensor.h"

namespace llm_system {
/// @brief 记录模块进入/退出执行图，并调用虚函数 forward；模型构图时用它建立层级与张量依赖。
Tensor::Ptr Module::operator()(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr return_tensor;
  device->top_module_graph->push_module_graph(getptr(), input);
  // operator() 包装虚函数调用，把模块嵌套结构记录到图中；forward 的具体行为由派生类决定。
  return_tensor = forward(input, sequences_metadata);
  device->top_module_graph->pop_module_graph(return_tensor);
  return return_tensor;
};

/// @brief 记录模块进入/退出执行图，并调用虚函数 forward；模型构图时用它建立层级与张量依赖。
TensorVec Module::operator()(const TensorVec input,
                             BatchedSequence::Ptr sequences_metadata) {
  std::vector<Tensor::Ptr> return_tensor;
  device->top_module_graph->push_module_graph(getptr(), input);
  return_tensor = forward(input, sequences_metadata);
  device->top_module_graph->pop_module_graph(return_tensor);
  return return_tensor;
};

/// @brief 递归更新当前模块所持张量和子模块的设备关联。
void Module::set_device(Device::Ptr device) {
  for (auto iter : tensor_list) {
    iter.second->set_device(device);
  }
  for (auto iter : module_list) {
    iter.second->set_device(device);
  }
};

/// @brief 按指定 weight/act/cache 标签汇总模块及子模块的张量字节数，用于容量估算。
long Module::size(std::vector<std::string> tag) {
  long act = 0;
  long weight = 0;
  long cache = 0;

  auto size_vector = get_size();
  act = size_vector.at(0);
  weight = size_vector.at(1);
  cache = size_vector.at(2);

  long total = 0;
  for (std::string _tag : tag) {
    if (!_tag.compare("act")) {
      total += act;
    } else if (!_tag.compare("weight")) {
      total += weight;
    } else if (!_tag.compare("cache")) {
      total += cache;
    }
  }
  return total;
}

/// @brief 递归汇总当前模块与子模块的权重、激活和缓存大小，返回分类字节数。
std::vector<long> Module::get_size() {
  long act = 0;
  long weight = 0;
  long cache = 0;
  for (auto _tensor : tensor_list) {
    Tensor::Ptr tensor = _tensor.second;
    if (tensor->tag == "act") {
      act += tensor->getSize();
    } else if (tensor->tag == "weight") {
      weight += tensor->getSize();
    } else if (tensor->tag == "cache") {
      cache += tensor->getSize();
    }
  }

  for (auto _module : module_list) {
    Ptr module = _module.second;
    auto size_vector = module->get_size();
    act += size_vector.at(0);

    // this line is for minimizing activation size. (calculate only the size of the largest activation per module)
    // act = std::max(act, size_vector.at(0));

    weight += size_vector.at(1);
    cache += size_vector.at(2);
  }

  std::vector<long> size_vector;
  size_vector.push_back(act);
  size_vector.push_back(weight);
  size_vector.push_back(cache);

  return size_vector;
}

/// @brief 把张量按名称加入本模块的 tensor_list，供 forward 和容量统计使用。
void Module::add_tensor(Tensor::Ptr tensor) {
  add_tensor(tensor->name, tensor);
}

/// @brief 把张量按名称加入本模块的 tensor_list，供 forward 和容量统计使用。
void Module::add_tensor(std::string name, Tensor::Ptr tensor) {
  if (tensor_list.find(name) == tensor_list.end()) {
    tensor_list.emplace(name, tensor);
  } else {
    tensor_list[name] = tensor;
  }
}

/// @brief 清除模块张量的就绪状态及相关本轮标记，供执行图重启使用。
void Module::unset_tensor() {
  Tensor::Ptr tensor;
  Module::Ptr module;
  for (auto _tensor : tensor_list) {
    tensor = _tensor.second;
    tensor->unset();
  }

  for (auto _module : module_list) {
    module = _module.second;
    module->unset_tensor();
  }
}

/// @brief 通过集群查找参与设备的对应张量，填充本模块相关的依赖列表。
void Module::set_dependency_tensor(
    std::vector<Tensor::Ptr>& dependency_tensor_list, Tensor::Ptr tensor) {
  device->cluster->set_dependency_tensor(dependency_tensor_list, tensor,
                                         device_list);
}

/// @brief 查找已有激活张量，按需要更新其形状/内存大小并设置就绪标记；不计算数值内容。
Tensor::Ptr Module::get_activation(std::string name, std::vector<int> shape,
                                   bool ready) {
  if (auto tensor = tensor_list.find(name); tensor != tensor_list.end()) {
    Tensor::Ptr return_tensor = tensor->second;
    return_tensor->shape = shape;
    return_tensor->setMemoryObject();
    if (ready) {
      return_tensor->set();
    };
    return return_tensor;
  } else {
    std::cout << name <<std::endl;
    fail("Unvalid activation tensor request");
    return nullptr;
  }
}

/// @brief 按缓存名、序列/头编号和压缩模式查找缓存张量，并重建其内存对象；当前 shape 参数未使用，不在此更新缓存形状。
Tensor::Ptr Module::get_cache(std::string name, int seq_idx, int kv_idx, bool compressed_kv,
                              std::vector<int> shape) {
  if(compressed_kv){
    name = name + "_" + std::to_string(seq_idx);
  }
  else{
    name = name + "_" + std::to_string(seq_idx) + "_" + std::to_string(kv_idx);
  }
  if (auto tensor = tensor_list.find(name); tensor != tensor_list.end()) {
    Tensor::Ptr return_tensor = tensor->second;
    return_tensor->setMemoryObject();
    return return_tensor;
  } else {
    std::cout << name << std::endl;
    fail("Unvalid cached tensor request");
    return nullptr;
  }
}

  /// @brief 把当前模块的共享指针写入它持有的张量，建立张量到所属模块的反向关联。
  void Module::set_tensor_module() {
    Tensor::Ptr tensor;
    for (auto& _tensor : tensor_list) {
      tensor = _tensor.second;
      tensor->set_module(getptr());
    }
  }

  /// @brief 保存命名子模块并向设备/集群登记模块，供图执行和跨设备依赖查找。
  void Module::add_module_with_name(std::string name, Module::Ptr module) {
    if (module_list.find(name) == module_list.end()) {
      module_list.emplace(name, module);
    } else {
      module_list[name] = module;
    }
    device->add_module(module->module_map_name, module);
  }

  }  // namespace llm_system
