// 中文导读：模型结构与设备映射：组合算子模块、配置维度和并行组，在初始化阶段建立执行图。
#pragma once
#include <map>
#include <string>

#include "common/assert.h"
#include "hardware/cluster.h"
#include "model/llm.h"
#include "model/model_config.h"
#include "module/module.h"
#include "scheduler/scheduler.h"

namespace llm_system {

/// @brief 模型构建入口：为各设备创建 LLM 模块，用初始 batch 元数据建立张量描述和执行图。
class Model : public Module {
 public:
  /// @brief 为每台设备创建 LLM，执行一次用于构图的形状传播，然后清空初始化阶段统计。
  Model(ModelConfig& model_config, Cluster::Ptr cluster,
        Scheduler::Ptr scheduler)
      : Module(),
        model_config(model_config),
        cluster(cluster),
        scheduler(scheduler) {
    for (int i = 0; i < cluster->num_total_device; i++) {
      device = cluster->get_device(i);
      device->setModelConfig(model_config);
      Module::Ptr top_module = std::static_pointer_cast<Module>(
          LLM::Create(model_config, cluster, scheduler, device));

      add_module(top_module);

      model_distribute(top_module, cluster, scheduler, i);
      device->reset_timeboard();
      device->reset_status();
    }
  };

 private:
  /// @brief 取得初始化用的最大 batch 描述和占位 embedding 输入，调用顶层模块以建立设备执行图。
  void model_distribute(Module::Ptr top_module, Cluster::Ptr cluster,
                        Scheduler::Ptr scheduler, int device_rank) {
    BatchedSequence::Ptr max_metadata = scheduler->getMaxMetadata(model_config.num_routed_expert, model_config.top_k);
    // Tensor::Ptr input_tensor = Tensor::Create(
    //     "EmbeddingVector", {1, max_metadata->get_process_token()}, "act",
    //     cluster->get_device(device_rank));


    Tensor::Ptr input_tensor = Tensor::Create(
          "EmbeddingVector", {1, scheduler->model_config.n_vocab}, "act",
          cluster->get_device(device_rank), scheduler->model_config.precision_byte);
    // Tensor::Ptr input_tensor =
    //     Tensor::Create("EmbeddingVector", {64, model_config.hidden_dim},
    //     "act",
    //                    cluster->get_device(device_rank));
    (*top_module)(input_tensor, max_metadata);
  };

  Cluster::Ptr cluster;
  Scheduler::Ptr scheduler;
  ModelConfig model_config;
};

}  // namespace llm_system