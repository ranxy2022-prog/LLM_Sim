// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/module_graph.h"

namespace llm_system {

/// @brief 保存模块、输入、统计板引用及层级/退出标志，初始化本轮时间戳标记。
ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         Tensor::Ptr input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = false;
  stamped = false;
};

/// @brief 保存模块、输入、统计板引用及层级/退出标志，初始化本轮时间戳标记。
ModuleGraph::ModuleGraph(Module::Ptr module, StatusBoard& status,
                         TensorVec input, int module_level, bool module_pop)
    : module(module),
      status(status),
      input_vec(input),
      module_level(module_level),
      module_pop(module_pop) {
  isTensorVec = true;
  stamped = false;
};

/// @brief 非执行记录直接通过；可执行模块在依赖就绪后调用 forward，否则返回 false 让设备暂时停止推进。
bool ModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  if (module == nullptr || !module->execution()) {
    return true;
  } else if (check_ready() == true) {
    if (isTensorVec) {
      module->forward(input_vec, sequences_metadata);
    } else {
      module->forward(input, sequences_metadata);
    }
    // input->unset();
    return true;
  } else {
    return false;
  }
};

/// @brief 建立该图记录的张量依赖列表。
void ModuleGraph::set_dependency() { set_dependency_tensor(); }

/// @brief 检查 dependency_tensor_list 中所有张量是否就绪；当前实现没有使用传入的 tensor_list 参数。
bool ModuleGraph::checkListReady(TensorVec tensor_list) {
  for (Tensor::Ptr tensor : dependency_tensor_list) {
    if (tensor->ready == false) {
      return false;
    }
  }
  return true;
}

/// @brief 同步模块等待跨设备依赖并对齐时间，普通模块检查输入就绪状态，决定能否执行。
bool ModuleGraph::check_ready() {
  if (module->sync) {
    if (!checkListReady(dependency_tensor_list)) {
      return false;
    }
    // all operations are doned, we have to sync the devices
    sync_devices();
    return true;
  } else {
    if (input && input->ready) {
      return true;
    } else if (input_vec.size() != 0) {
      if (checkListReady(input_vec)) {
        return true;
      }
    }
  }
  return false;
}

/// @brief 根据同步依赖将相关设备时间线对齐，避免消费者早于生产者完成时刻执行。
void ModuleGraph::sync_devices() {
  // not yet synced
  if (input) {
    if (!input->timeboard_synced) {
      Device::Ptr device;
      time_ns time = 0;
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        device = tensor->get_device();
        time_ns device_time = device->get_time();
        time = std::max(time, device_time);
      }
      for (Tensor::Ptr tensor : dependency_tensor_list) {
        tensor->timeboard_synced = true;
        device = tensor->get_device();
        device->set_time(time);
      }
      input->timeboard_synced = false;
    } else {
      input->timeboard_synced = false;
    }
  } else {
    fail("Module cannot be synced when it's inputs are TensorVector");
  }
}

/// @brief 为需要跨设备同步的模块取得对应依赖张量，供 ready 检查和时间同步。
void ModuleGraph::set_dependency_tensor() {
  // only when inputs are one Tensor pointer
  if (input && module && module->sync) {
    module->set_dependency_tensor(dependency_tensor_list, input);
  }
}

/// @brief 打印当前执行记录的模块名称、层级和形状等调试信息。
void ModuleGraph::print_graph() {
  if (module != nullptr) {
    for (int i = 0; i < module_level; i++) {
      std::cout << "\t";
    }
    std::cout << module->name << std::endl;
  }
}

/// @brief 关联设备统计板并初始化执行记录容器和时间戳管理状态。
TopModuleGraph::TopModuleGraph(StatusBoard& status)
    : status(status), module_graph(){};

/// @brief 在构图时追加模块进入记录，保存单张量或张量列表输入，并增加层级。
void TopModuleGraph::push_module_graph(Module::Ptr module, Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

/// @brief 在构图时追加模块退出记录，保存输出描述并恢复父层级。
void TopModuleGraph::pop_module_graph(Tensor::Ptr input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

/// @brief 在构图时追加模块进入记录，保存单张量或张量列表输入，并增加层级。
void TopModuleGraph::push_module_graph(Module::Ptr module, TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, module, input, current_module_level++, false);
  module_graph.push_back(graph);
};

/// @brief 在构图时追加模块退出记录，保存输出描述并恢复父层级。
void TopModuleGraph::pop_module_graph(TensorVec input) {
  ModuleGraph::Ptr graph =
      ModuleGraph::Create(status, nullptr, input, current_module_level--, true);
  module_graph.push_back(graph);
};

/// @brief 遍历执行记录，建立需要的模块/跨设备张量依赖。
void TopModuleGraph::set_dependency() {
  for (auto module : module_graph) {
    module->set_dependency();
  }
  restart_graph();
}

/// @brief 从当前迭代器顺序执行记录并采集时间戳；遇到未满足的依赖即暂停，供集群稍后重试。
void TopModuleGraph::run(BatchedSequence::Ptr sequences_metadata) {
  for (; current_module != module_graph.end(); current_module++) {
    set_stamp();
    // if execution is blocked because of sync
    if (!(*current_module)->run(sequences_metadata)) {
      break;
    }
  }
}

// deprecated
/// @brief 已废弃接口：当前调用会触发 fail，正常路径使用 set_stamp。
void TopModuleGraph::push_stamp() {
  fail("Cannot use push_stamp function");
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_pop()) {
    timeboard.push_timestamp(status, module_graph->get_name());
  }
}

// deprecated
/// @brief 已废弃接口：当前调用会触发 fail，正常路径使用 set_stamp。
void TopModuleGraph::pop_stamp() {
  fail("Cannot use pop_stamp function");
  ModuleGraph::Ptr module_graph = *current_module;
  if (module_graph->is_pop()) {
    timeboard.pop_timestamp(status);
  }
}

/// @brief 根据进入/退出记录采集时间戳；用 stamped 标志避免依赖阻塞重试时重复累计。
void TopModuleGraph::set_stamp() {
  ModuleGraph::Ptr module_graph = *current_module;
  if (!module_graph->is_stamped()) {
    if (module_graph->is_pop()) {
      set_pop_status();
      timeboard.pop_timestamp(status);
    } else {
      set_push_status();
      timeboard.push_timestamp(status, module_graph->get_name());
    }
    module_graph->set_stamped();
  }
}

/// @brief 进入模块前更新统计板的输入形状与异构时间线状态，作为时间戳起始快照。
void TopModuleGraph::set_push_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;

    status.device_time = std::max(status.device_time,
                                  std::max(status.low_time, status.high_time));
    status.low_time = status.device_time;
    status.high_time = status.device_time;
    status.parallel_execution = false;
    //

  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
      if (status.tensor->isPerformHigh()) {
        status.device_time = status.high_time;
      } else {
        status.device_time = status.low_time;
      }
    } else {
      status.device_time = std::max(
          status.device_time, std::max(status.low_time, status.high_time));
      status.low_time = status.device_time;
      status.high_time = status.device_time;
      status.parallel_execution = false;
    }
  }
}

/// @brief 取走算子 ExecStatus，累计 device/high/low 时间及命令/计算能耗，更新输出形状统计。
void TopModuleGraph::set_pop_status() {
  if ((*current_module)->isTensorVec) {
    status.isTensorVec = true;
    status.tensor_vec = (*current_module)->input_vec;
  } else {
    status.isTensorVec = false;
    status.tensor = (*current_module)->input;
    if (status.tensor->parallel_execution) {
      status.parallel_execution = true;
    } else {
      status.parallel_execution = false;
    }
  }

  ExecStatus exec_status = device->getExecStatus();

  if (status.parallel_execution) {
    if (exec_status.processor_type == ProcessorType::LOGIC ||
        exec_status.processor_type == ProcessorType::PIM) {
      status.low_time += exec_status.total_duration;
      status.device_time = status.low_time;
    } else if (exec_status.processor_type == ProcessorType::GPU) {
      status.high_time += exec_status.total_duration;
      status.device_time = status.high_time;
    }
  } else {
    status.device_time += exec_status.total_duration;
    // status.high_time = status.device_time;
    // status.low_time = status.device_time;
  }
  
  if (exec_status.processor_type == ProcessorType::PIM ||
      exec_status.processor_type == ProcessorType::LOGIC ||
      exec_status.processor_type == ProcessorType::GPU) {
    int processor_type = (int)exec_status.processor_type;
    status.act_energy +=
        exec_status.act_count * dram_powers[processor_type].kACT_energy_j_;
    status.read_energy +=
        exec_status.read_count * dram_powers[processor_type].kREAD_energy_j_;
    status.write_energy +=
        exec_status.write_count * dram_powers[processor_type].kWRITE_energy_j_;

    status.all_act_energy += exec_status.all_act_count *
                             dram_powers[processor_type].kALL_ACT_energy_j_;
    status.all_read_energy += exec_status.all_read_count *
                              dram_powers[processor_type].kALL_READ_energy_j_;
    status.all_write_energy += exec_status.all_write_count *
                               dram_powers[processor_type].kALL_WRITE_energy_j_;

    status.mac_energy +=
        exec_status.flops * dram_powers[processor_type].kMAC_energy_j_;
    ;  // 2flops per operation, energy per operation, pJ to nJ
  }

  // if (!exec_status.parallel_execution) {
  //   // status.device_time = std::max(status.device_time,
  //   //                               std::max(status.low_time,
  //   //                               status.high_time));
  //   // status.low_time = status.device_time;
  //   // status.high_time = status.device_time;
  //   status.parallel_execution = false;
  // } else {
  //   status.parallel_execution = true;
  // }

  // if (exec_status.processor_type == ProcessorType::LOGIC ||
  //     exec_status.processor_type == ProcessorType::PIM) {
  //   status.low_time += exec_status.total_duration;
  //   status.device_time = status.low_time;
  // } else if (exec_status.processor_type == ProcessorType::GPU) {
  //   status.high_time += exec_status.total_duration;
  //   status.device_time = status.high_time;
  // } else {
  //   status.low_time = status.device_time;
  //   status.high_time = status.device_time;
  // }

  status.compute_util = exec_status.compute_util;
  status.memory_util = exec_status.memory_util;
  status.processor_type = exec_status.processor_type;

  status.flops += exec_status.flops;
  status.memory_size += exec_status.memory_size;

  status.opb = exec_status.opb;
}

/// @brief 遍历并打印设备执行图记录，帮助检查模型构建顺序。
void TopModuleGraph::print_graph() {
  std::cout << "Print graph" << std::endl;
  for (auto module_graph_ : module_graph) {
    module_graph_->print_graph();
  }
}

/// @brief 登记某一处理器类型的命令能耗系数。
void TopModuleGraph::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  if(dram_powers.size() == 0){
    for (int i = 0; i < (int)ProcessorType::MAX; i++) {
      DramEnergy temp;
      dram_powers.push_back(temp);
    }
  }
  dram_powers[ProcessorType] = dramEnergy;
}

/// @brief 汇总设备 DRAM 各类命令能耗与计算能耗，返回分类数组。
std::vector<energy_nJ> TopModuleGraph::getDeviceEnergy(){
  std::vector<energy_nJ> device_energy {status.act_energy, status.read_energy, status.write_energy, 
                            status.all_act_energy, status.all_read_energy, status.all_write_energy,
                            status.mac_energy, status.act_energy + status.read_energy + status.write_energy + 
                            status.all_act_energy + status.all_read_energy + status.all_write_energy + status.mac_energy}; 
  return device_energy;
}

/// @brief 回到执行序列起点，清除各记录时间戳和张量就绪状态，为下一轮回放做准备。
void TopModuleGraph::restart_graph() {
  current_module = module_graph.begin();
  assertTrue(current_module != module_graph.end(),
             "No module in TopModuleGraph");
  (*current_module)->set_ready();

  for (auto module_graph_ : module_graph) {
    module_graph_->unset_tensor();
  }
};

}  // namespace llm_system