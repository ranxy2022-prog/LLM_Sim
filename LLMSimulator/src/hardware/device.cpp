// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#include "hardware/device.h"

#include <random>
#include <string>
#include <stdexcept>

#include "common/assert.h"
#include "dram/dram_interface.h"
#include "dram/dram_request.h"
#include "dram/mmap_controller.h"
#include "dram/pimkernel/pim_kernel.h"
#include "module/module_graph.h"
#include "module/tensor.h"

namespace llm_system {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_int_distribution<int> dis(0, 5);

/// @brief 保存设备配置/rank，创建执行图、地址映射器和 Ramulator 接口；初始关闭正式性能估算。
Device::Device(SystemConfig config, int device_total_rank, Cluster_ptr cluster)
    : config(config),
      device_total_rank(device_total_rank),
      cluster(cluster),
      status() {
  compute_peak_flops = config.compute_peak_flops;
  memory_bandwidth = config.memory_bandwidth;
  memory_capacity = config.memory_capacity;
  device_local_rank = device_total_rank % config.num_device;

  top_module_graph = TopModuleGraph::Create(status);

  std::string dram_cfg_path;
  if(config.gpu_gen == "H100"){
    dram_cfg_path = "./dram_config_HBM3_80GB.yaml";
  }
  else if(config.gpu_gen == "B100" || config.gpu_gen == "B200"){
    dram_cfg_path = "./dram_config_HBM3E_192GB.yaml";
  }
  YAML::Node cfg = YAML::LoadFile(dram_cfg_path);

  double memory_scale_factor = 0;
  MemoryConfig memory_config = MemoryConfig(config.num_cube, config.num_logic_cube);
  if(config.gpu_gen == "H100"){
    // H100, HBM3 80GB, 5.2Gbps
    memory_scale_factor = 0.76923;
    memory_config = hbm3_80GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }
  else if((config.gpu_gen == "B100") || (config.gpu_gen == "B200")){
    // B100, HBM3E 192GB, 8Gbps
    memory_scale_factor = 0.5; // 8.0Gbps pin rate's ideal bandwidth = 8000, tCK = 2000, 1 / 2GHz = 0.5
    memory_config = hbm3e_192GB;
    memory_config.num_cube = config.num_cube;
    memory_config.num_logic_cube = config.num_logic_cube;
  }

  dram_interface = DRAMInterface::Create(dram_cfg_path, memory_scale_factor);
  mmap_controller = MMapController::Create(memory_config);
  use_ramulator = config.use_ramulator;
  perform_execution = false;
}

/// @brief 把依赖建立工作委托给本设备执行图。
void Device::set_dependency() { top_module_graph->set_dependency(); }

/// @brief 查询本设备执行图是否还有未处理记录。
bool Device::check_module_graph_remain() {
  return top_module_graph->check_module_graph_remain();
};

/// @brief 按全局设备 rank 和非专家 TP 数找到 DP batch，使用该 batch 推进本设备执行图。
void Device::run(std::vector<BatchedSequence::Ptr> sequences_metadata_list) {
  int dp_rank = device_total_rank / model_config.ne_tp_dg;
  top_module_graph->run(sequences_metadata_list.at(dp_rank));
}

/// @brief 把本设备执行图回到起点，重置张量就绪/时间戳状态以便下一轮执行。
void Device::restartGraph() { top_module_graph->restart_graph(); }

/// @brief 把自身共享指针交给执行图，使执行图可以读取算子结果并更新设备状态。
void Device::connectTopModuleGraph() {
  top_module_graph->connectDevice(get_ptr());
}

/// @brief 用默认 StatusBoard 重置设备的本轮累计时间、能耗等状态。
void Device::reset_status() { status = StatusBoard(); }
/// @brief 清空模块级时间戳统计树，为新一轮记录做准备。
void Device::reset_timeboard() { top_module_graph->reset_timeboard(); }

/// @brief 把模块及其名称转交 Cluster，登记到本设备的模块索引表。
void Device::add_module(std::string name, Module_ptr module) {
  cluster->add_module(device_total_rank, name, module);
}

/// @brief 把张量交给 MMapController，建立模拟内存地址对象；不分配实际 GPU 显存。
void Device::setMemoryObject(Tensor::Ptr tensor) {
  mmap_controller->setMemoryObject(tensor);
}

/// @brief 把性能结果按缓存键写入集群级缓存；键由算子、处理器、请求类型和数据大小等组成。
void Device::addExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

/// @brief 把性能结果按缓存键写入集群级缓存；键由算子、处理器、请求类型和数据大小等组成。
void Device::addExecutionCache(ExecStatus& exec_status, LayerType layer_type,
                               ProcessorType processor_type,
                               DRAMRequestType dram_reqeust_type, long size) {
  CacheKey key =
      std::make_tuple(layer_type, processor_type, dram_reqeust_type, size);
  auto& cache = cluster->execution_time_cache;
  cache.emplace(key, exec_status);
}

/// @brief 查询执行结果缓存，命中时复制到输出状态或 DRAM 接口状态；缓存不描述完整地址/行缓冲历史。
bool Device::checkExecutionCache(ExecStatus& exec_status, CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    exec_status = (*cache_iter).second;
    return true;
  } else {
    return false;
  }
}

/// @brief 查询执行结果缓存，命中时复制到输出状态或 DRAM 接口状态；缓存不描述完整地址/行缓冲历史。
bool Device::checkExecutionCache(CacheKey key) {
  auto& cache = cluster->execution_time_cache;
  if (const auto& cache_iter = cache.find(key); cache_iter != cache.end()) {
    ExecStatus exec_status = (*cache_iter).second;
    ExecStatus& _status = dram_interface->getExecStatus();
    _status = exec_status;
    return true;
  } else {
    return false;
  }
}

/// @brief 保存当前算子的执行结果，供执行图在模块退出时累计。
void Device::setExecStatus(ExecStatus& exec_status_) {
  exec_status = exec_status_;
}

/// @brief 取走高算力单元的暂存结果并清零该结果槽。
ExecStatus Device::getHighExecStatus() {
  ExecStatus return_status = high_exec_status;
  high_exec_status = ExecStatus();
  return return_status;
}

/// @brief 取走低算力单元的暂存结果并清零该结果槽。
ExecStatus Device::getLowExecStatus() {
  ExecStatus return_status = low_exec_status;
  low_exec_status = ExecStatus();
  return return_status;
}

/// @brief 取走当前算子的执行结果，并将该暂存状态清零，避免重复累计。
ExecStatus Device::getExecStatus() {
  ExecStatus return_status = exec_status;
  exec_status = ExecStatus();
  return return_status;
}

// check whether execution must be performed, and ramulator
/// @brief 正式执行开关开启时清空内存计数并调用集群 Executor，按算子类型估计本轮成本。
void Device::execution(LayerType layer_type,
                       const std::vector<Tensor::Ptr>& tensor_list,
                       const BatchedSequence::Ptr sequences_metadata,
                       const LayerInfo layer_info) {
  if (perform_execution) {
    dram_interface->resetCounter();
    cluster->executor.execution(layer_type, tensor_list, sequences_metadata,
                                config.processor_type, layer_info,
                                use_ramulator, get_ptr());
  }
}

/// @brief 旧调试入口：当前实现只取得输入/权重/输出并打印权重名称，未在这里发起完整仿真。
void Device::execution_ramulator(LayerType layer_type,
                                 std::vector<Tensor::Ptr> tensor_list) {
  Tensor::Ptr input = tensor_list.at(0);
  Tensor::Ptr weight = tensor_list.at(1);
  Tensor::Ptr output = tensor_list.at(2);

  std::cout << weight->name << std::endl;
}

/// @brief 把一个逻辑 DRAM 请求包装成列表，交给 DRAMInterface 执行。
void Device::run_ramulator(DRAMRequest_Ptr dram_request) {
  std::list<DRAMRequest::Ptr> request;
  request.push_back(dram_request);
  dram_interface->HandleRequest(request, 0);
}

/// @brief 按张量字节数、访问粒度、cube/伪通道数量及每行列数估算 ACT/读写次数；不遍历访问地址，也不模拟行冲突时序。
void Device::run_ideal(DRAMRequestType dram_request_type, Tensor_Ptr tensor){
  long total_size = tensor->getSize(); // Byte
  if (total_size == 0) {
    return;
  }
  MemoryConfig memory_config = mmap_controller->getConfig();
  int num_cube = memory_config.num_cube;
  int num_channel = memory_config.num_channel; // 32 (not legacy, pCH)
  int num_col = memory_config.num_col;
  int granul = mmap_controller->getGranul();

  long total_read = total_size / granul;
  long rw_cmd_to_cube_0 = (total_read % num_cube == 0) ? (total_read / num_cube) : ((total_read / num_cube) + 1);

  long rw_cmd_to_pCH_0 = (rw_cmd_to_cube_0 % num_channel == 0) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);
  long rw_cmd_to_pCH_1 = (rw_cmd_to_cube_0 % num_channel == 1) ? (rw_cmd_to_cube_0 / num_channel) : ((rw_cmd_to_cube_0 / num_channel) + 1);

  dram_interface->resetCounter();
  dram_interface->getExecStatus().act_count = (((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) < num_col) ? 1 : ((rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1) / num_col));
  if(dram_request_type == DRAMRequestType::kRead){
    dram_interface->getExecStatus().read_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
  else if(dram_request_type == DRAMRequestType::kWrite){
    dram_interface->getExecStatus().write_count = (rw_cmd_to_pCH_0 + rw_cmd_to_pCH_1);
  }
}

/// @brief 向本设备执行图登记处理器对应的单次命令/运算能耗参数。
void Device::initializeDRAM(int ProcessorType, DramEnergy dramEnergy) {
  int num_pseudo_ch = 0;
  if (ProcessorType == (int)(ProcessorType::GPU)) {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel / 2;
  }
  else {
    num_pseudo_ch = mmap_controller->getConfig().num_cube * mmap_controller->getConfig().num_channel;
  }
  dramEnergy.kACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kREAD_energy_j_ *= num_pseudo_ch;
  dramEnergy.kWRITE_energy_j_ *= num_pseudo_ch;

  dramEnergy.kALL_ACT_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_READ_energy_j_ *= num_pseudo_ch;
  dramEnergy.kALL_WRITE_energy_j_ *= num_pseudo_ch;

  top_module_graph->initializeDRAM(ProcessorType, dramEnergy);
}

};  // namespace llm_system