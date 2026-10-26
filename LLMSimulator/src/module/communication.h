// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>

#include "module/module.h"
#include "module/tensor.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 集体归约性能模块：按环式通信跳数、固定延迟与数据量/带宽估算时间。
class AllReduce : public Module {
 public:
  using Ptr = std::shared_ptr<AllReduce>;

  /// @brief AllReduce 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllReduce(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 AllReduce 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  AllReduce(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief AllReduce 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  AllReduce() = default;

  /// @brief 以 2*(设备数-1) 跳和每跳固定延迟+分片字节数/带宽估算环式归约，累加设备时间。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 集合通信占位模块：当前 forward 只保持输出形状，未独立估算 all-gather 传输开销。
class AllGather : public Module {
 public:
  using Ptr = std::shared_ptr<AllGather>;

  /// @brief AllGather 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllGather(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 AllGather 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  AllGather(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief AllGather 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  AllGather() = default;

  /// @brief 取得与输入同形状的输出描述；当前实现未计算 all-gather 的传输时间。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};
 
/// @brief 分发通信占位模块：当前 forward 按参与设备数缩小特征维度，未独立估算通信开销。
class AllScatter : public Module { // or All-to-All
 public:
  using Ptr = std::shared_ptr<AllScatter>;

  /// @brief AllScatter 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new AllScatter(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 AllScatter 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  AllScatter(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief AllScatter 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  AllScatter() = default;

  /// @brief 将输入特征维度按参与设备数拆分并返回输出描述；当前没有单独通信计时。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
  int parallel_num;
};

/// @brief MoE token 分发模块：根据专家分布估计节点内/节点间通信量和耗时。
class MoEScatter : public Module {
 public:
  using Ptr = std::shared_ptr<MoEScatter>;

  /// @brief MoEScatter 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEScatter(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 MoEScatter 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  MoEScatter(std::string& prefix, std::string& name,
             std::vector<int> device_list, Device::Ptr device);
  /// @brief MoEScatter 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  MoEScatter() = default;

  /// @brief 依据专家 token 分布划分节点内/节点间通信量，按互连参数估算分发时间并更新设备状态。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief MoE 专家输出收集模块：构造汇聚输出并估计节点内/节点间通信开销。
class MoEGather : public Module {
 public:
  using Ptr = std::shared_ptr<MoEGather>;

  /// @brief MoEGather 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new MoEGather(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 MoEGather 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  MoEGather(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief MoEGather 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  MoEGather() = default;

  /// @brief 收集专家输出描述，按专家负载与互连参数估算汇聚通信时间。
  TensorVec forward(const TensorVec input,
                    BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 同步组合模块：先合并本设备高低处理器时间线，再建立跨设备同步依赖。
class Sync : public Module {
 public:
  using Ptr = std::shared_ptr<Sync>;

  /// @brief Sync 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Sync 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Sync(std::string& prefix, std::string& name, std::vector<int> device_list,
       Device::Ptr device);
  /// @brief Sync 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Sync() = default;

  /// @brief 先执行设备内高低时间线汇合，再通过同步子模块建立跨设备等待关系。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 设备内时间线汇合模块：把 device/high/low 时间推进到三者最大值。
class Sync__Set : public Module {
 public:
  using Ptr = std::shared_ptr<Sync__Set>;

  /// @brief Sync__Set 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync__Set(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Sync__Set 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Sync__Set(std::string& prefix, std::string& name,
            std::vector<int> device_list, Device::Ptr device);
  /// @brief Sync__Set 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Sync__Set() = default;

  /// @brief 将 device_time、high_time、low_time 对齐到最大值，并返回同步标记张量。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

/// @brief 跨设备同步标记模块：依赖等待和时间对齐由执行图负责，forward 直接传递输入。
class Sync__ : public Module {
 public:
  using Ptr = std::shared_ptr<Sync__>;

  /// @brief Sync__ 的静态工厂：调用构造函数并建立 shared_ptr 管理，随后建立张量到模块的关联；因此需要先建立 shared_ptr。
  [[nodiscard]] static Ptr Create(std::string prefix, std::string name,
                                  std::vector<int> device_list,
                                  Device::Ptr device) {
    Ptr ptr = Ptr(new Sync__(prefix, name, device_list, device));
    ptr->set_tensor_module();
    return ptr;
  };

 private:
  /// @brief 初始化 Sync__ 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
  Sync__(std::string& prefix, std::string& name, std::vector<int> device_list,
         Device::Ptr device);
  /// @brief Sync__ 的默认/移动构造，按成员自身规则初始化；不额外执行模型计算。
  Sync__() = default;

  /// @brief 直接返回输入；该模块的等待与跨设备时间对齐由 ModuleGraph 的 sync 分支处理。
  Tensor::Ptr forward(const Tensor::Ptr input,
                      BatchedSequence::Ptr sequences_metadata) override;
};

}  // namespace llm_system