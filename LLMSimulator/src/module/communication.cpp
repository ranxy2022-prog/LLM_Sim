// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
#include "module/communication.h"
#include "scheduler/scheduler.h"

#include "common/assert.h"
// AllReduce //

namespace llm_system {

/// @brief 初始化 AllReduce 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AllReduce::AllReduce(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) { // sync == true, thererfore need to be synced
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output = Tensor::Create("allreduce_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 以 2*(设备数-1) 跳和每跳固定延迟+分片字节数/带宽估算环式归约，累加设备时间。
Tensor::Ptr AllReduce::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];

  Tensor::Ptr output = get_activation("allreduce_output", input->shape);

  long size = input->getSize();
  if (size == 0) {
    return output;
  }

  int hop = (device_list.size() - 1) * 2;
  size /= device_list.size();

  time_ns one_hop =
      device->config.device_ict_latency +
      size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000;

  time_ns total_time = one_hop * hop;

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

/// @brief 初始化 AllGather 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AllGather::AllGather(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output = Tensor::Create("allgather_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 取得与输入同形状的输出描述；当前实现未计算 all-gather 的传输时间。
Tensor::Ptr AllGather::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {

  Tensor::Ptr output = get_activation("allgather_output", input->shape);
  return output;
}

/// @brief 初始化 AllScatter 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
AllScatter::AllScatter(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  parallel_num = device_list.size();

  Tensor::Ptr output = Tensor::Create("allscatter_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 将输入特征维度按参与设备数拆分并返回输出描述；当前没有单独通信计时。
Tensor::Ptr AllScatter::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  long m = input->shape[0];
  long k = input->shape[1];

  std::vector<int> shape = {input->shape[0], input->shape[1] / parallel_num};

  Tensor::Ptr output = get_activation("allscatter_output", shape);

  return output;
}

/// @brief 初始化 MoEScatter 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
MoEScatter::MoEScatter(std::string& prefix, std::string& name,
                       std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output =
      Tensor::Create("moe_scatter_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 依据专家 token 分布划分节点内/节点间通信量，按互连参数估算分发时间并更新设备状态。
Tensor::Ptr MoEScatter::forward(const Tensor::Ptr input,
                                BatchedSequence::Ptr sequences_metadata) {
  int m = input->shape[0];
  int k = input->shape[1];
  Tensor::Ptr output = get_activation("moe_scatter_output", input->shape);

  
  // assume using disaggregated system

  if (!device->perform_execution) {
    return output;
  }

  time_ns total_time = 0;
  time_ns send_time = 0;
  time_ns receive_time = 0;

  // Send time // 
  int intra_node_comm_token = 0;
  int inter_node_comm_token = 0;

  int src = device->device_total_rank; // current device
  int src_node = src / device->config.num_device;

  int ne_tp_dg = device->model_config.ne_tp_dg;
  int e_tp_dg = device->model_config.e_tp_dg;

  std::vector<int> tp_sharing_device_list = {};
  int device_list_offset = device->device_total_rank / ne_tp_dg * ne_tp_dg;

  for(int device_idx = device_list_offset; device_idx < device_list_offset + ne_tp_dg; device_idx ++){
    tp_sharing_device_list.push_back(device_idx);
  }

  std::unordered_set<int> set_tp_devices(tp_sharing_device_list.begin(), tp_sharing_device_list.end());

  int total_num_device = device->config.num_device * device->config.num_node;

  for(int dst = 0; dst < total_num_device; dst ++){ // dst: destination device
    if(set_tp_devices.count(dst) == 0){ // outer tp space
      int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (dst / e_tp_dg) * e_tp_dg;
      int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

      if((int)(dst / 8) == src_node){ 
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id]; // to the experts in a dst device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg (send only 1/ne_tp_dg tokens)
  
  hw_metric intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  hw_metric inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return output;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    send_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  // Receive time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int dst_device_rank = device->device_total_rank; // dst: destination device, current device
  int dst_node = dst_device_rank / device->config.num_device;

  int dst_dp_rank = device->device_total_rank / ne_tp_dg; // data parallel index
  
  int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (dst_device_rank / e_tp_dg) * e_tp_dg; // experts in a dst_device
  int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

  for(int src_dp_idx = 0; src_dp_idx < total_num_device / ne_tp_dg; src_dp_idx ++){ 
    if(src_dp_idx != dst_dp_rank){ // from other dp space

      int src_device_rank = src_dp_idx * ne_tp_dg;
      int src_node = src_device_rank / device->config.num_device;

      if(dst_node == src_node){ 
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->scheduler->running_queue[src_dp_idx]->local_num_token_in_expert[e_id]; // from the experts in a src device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->scheduler->running_queue[src_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg (send only 1/ne_tp_dg tokens)
  
  intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return output;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    receive_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  total_time = std::max(send_time, receive_time);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
      // device->status.device_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return output;
}

/// @brief 初始化 MoEGather 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
MoEGather::MoEGather(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {1, 1};

  Tensor::Ptr output =
      Tensor::Create("moe_gather_output", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 收集专家输出描述，按专家负载与互连参数估算汇聚通信时间。
TensorVec MoEGather::forward(const TensorVec input_vec,
                             BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr input = input_vec.at(0);
  int m = input->shape[0];
  int k = input->shape[1];
  Tensor::Ptr output = get_activation("moe_gather_output", input->shape);
  
  // assume using disaggregated system
  if (!device->perform_execution) {
    return input_vec;
  }

  time_ns total_time = 0;
  time_ns send_time = 0;
  time_ns receive_time = 0;

  // Receive time // 
  int intra_node_comm_token = 0;
  int inter_node_comm_token = 0;

  int dst = device->device_total_rank; // current device
  int dst_node = dst / device->config.num_device;

  int e_tp_dg = device->model_config.e_tp_dg;
  int ne_tp_dg = device->model_config.ne_tp_dg;

  std::vector<int> e_tp_sharing_device_list = {};
  int device_list_offset = device->device_total_rank / e_tp_dg * e_tp_dg;

  for(int device_idx = device_list_offset; device_idx < device_list_offset + e_tp_dg; device_idx ++){
    e_tp_sharing_device_list.push_back(device_idx);
  }

  std::unordered_set<int> set_e_tp_devices(e_tp_sharing_device_list.begin(), e_tp_sharing_device_list.end());

  int total_num_device = device->config.num_device * device->config.num_node;
  
  for(int src = 0; src < total_num_device; src ++){ // dst: destination device
    if(set_e_tp_devices.count(src) == 0){ // outer tp space
      int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (src / e_tp_dg) * e_tp_dg;
      int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

      if((int)(src / 8) == dst_node){ 
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id]; // to the experts in a src device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  hw_metric intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  hw_metric inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= device->model_config.e_tp_dg;
  inter_node_comm_size /= device->model_config.e_tp_dg;

  intra_node_comm_size /= device->model_config.ne_tp_dg; // receive only (1 / tp_degree) tokens, and then all reduce
  inter_node_comm_size /= device->model_config.ne_tp_dg; // receive only (1 / tp_degree) tokens, and then all reduce

  // FP8 dispatch && BF16 combine
  if((device->model_config.model_name == "deepseekV3") && device->model_config.precision_byte == 1){
    intra_node_comm_size *= 2;
    inter_node_comm_size *= 2;
  }

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return input_vec;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    receive_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      receive_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  // Send time //
  intra_node_comm_token = 0;
  inter_node_comm_token = 0;

  int src_device_rank = device->device_total_rank; // src: source device, current device
  int src_node = src_device_rank / device->config.num_device;

  int src_dp_rank = device->device_total_rank / ne_tp_dg; // data parallel index
  
  int expert_id_offset = device->model_config.num_routed_expert / total_num_device * (src_device_rank / e_tp_dg) * e_tp_dg; // experts in a src_device
  int num_expert_per_device = device->model_config.num_routed_expert / total_num_device * e_tp_dg;

  for(int dst_dp_idx = 0; dst_dp_idx < total_num_device / ne_tp_dg; dst_dp_idx ++){ 
    if(dst_dp_idx != src_dp_rank){ // to other dp space

      int dst_device_rank = dst_dp_idx * ne_tp_dg;
      int dst_node = dst_device_rank / device->config.num_device;

      if(dst_node == src_node){ 
        // intra node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          intra_node_comm_token += sequences_metadata->scheduler->running_queue[dst_dp_idx]->local_num_token_in_expert[e_id]; // to orig device from the experts in a src device
        }
      }
      else{ 
        // inter node
        for(int e_id = expert_id_offset; e_id < expert_id_offset + num_expert_per_device; e_id ++){
          inter_node_comm_token += sequences_metadata->scheduler->running_queue[dst_dp_idx]->local_num_token_in_expert[e_id];
        }
      }
    }
  }

  // if ne_tp_dg > 1, tp sharing devices have same tokens. Therefore, need to be divided by ne_tp_dg (send only 1/ne_tp_dg tokens)
  
  intra_node_comm_size = 1.0 * intra_node_comm_token * k * input->precision_byte;
  inter_node_comm_size = 1.0 * inter_node_comm_token * k * input->precision_byte;

  intra_node_comm_size /= ne_tp_dg;
  inter_node_comm_size /= ne_tp_dg;

  if(intra_node_comm_size == 0 && inter_node_comm_size == 0){
    return input_vec;
  }

  if(sequences_metadata->get_sum_process_token() > 0){
    // prefill & mixed stage - use both NVLink and InfiniBand

    time_ns intra_node_latency = intra_node_comm_size / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.device_ict_latency;
                                
    time_ns inter_node_latency = inter_node_comm_size / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
                                + device->config.node_ict_latency;

    send_time = std::max(intra_node_latency, inter_node_latency);
  }
  else if (sequences_metadata->get_gen_process_token() > 0){
    // decode - use only InifiniBand, but when num_node == 1, use NVLink
    if(device->config.num_node == 1){
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.device_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.device_ict_latency;
    }
    else{
      send_time = (intra_node_comm_size + inter_node_comm_size) / device->config.node_ict_bandwidth * 1000 * 1000 * 1000
      + device->config.node_ict_latency;
    }
  }

  total_time = std::max(send_time, receive_time);

  if (input->parallel_execution && !device->config.communication_hiding) {
    if (input->isPerformHigh()) {
      device->status.high_time += total_time;
    } else {
      device->status.low_time += total_time;
    }
  }
  device->status.device_time += total_time;

  return input_vec;
}

/// @brief 初始化 Sync 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
Sync::Sync(std::string& prefix, std::string& name, std::vector<int> device_list,
           Device::Ptr device)
    : Module(prefix, name, device, device_list) {
  std::vector<int> shape = {1, 1};

  auto sync__set =
      Sync__Set::Create(module_map_name, "sync__set", device_list, device);
  add_module(sync__set);

  auto sync__ = Sync__::Create(module_map_name, "sync__", device_list, device);
  add_module(sync__);
}

/// @brief 先执行设备内高低时间线汇合，再通过同步子模块建立跨设备等待关系。
Tensor::Ptr Sync::forward(const Tensor::Ptr input,
                          BatchedSequence::Ptr sequences_metadata) {
  auto sync__set = get_module("sync__set");
  auto sync__ = get_module("sync__");

  Tensor::Ptr temp = (*sync__set)(input, sequences_metadata);
  Tensor::Ptr output = (*sync__)(temp, sequences_metadata);
  return output;
}

/// @brief 初始化 Sync__Set 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
Sync__Set::Sync__Set(std::string& prefix, std::string& name,
                     std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true) {
  std::vector<int> shape = {1, 1};
  Tensor::Ptr output = Tensor::Create("sync", shape, "act", device, device->model_config.precision_byte);
  add_tensor(output);
}

/// @brief 将 device_time、high_time、low_time 对齐到最大值，并返回同步标记张量。
Tensor::Ptr Sync__Set::forward(const Tensor::Ptr input,
                               BatchedSequence::Ptr sequences_metadata) {
  Tensor::Ptr output = get_activation("sync");
  device->status.device_time =
      std::max(device->status.device_time,
               std::max(device->status.low_time, device->status.high_time));
  device->status.high_time = device->status.device_time;
  device->status.low_time = device->status.device_time;
  return output;
}

/// @brief 初始化 Sync__ 的配置成员，以及该模块所需的张量描述或子模块；创建对象不等于已经执行 forward。
Sync__::Sync__(std::string& prefix, std::string& name,
               std::vector<int> device_list, Device::Ptr device)
    : Module(prefix, name, device, device_list, true, true) {}

/// @brief 直接返回输入；该模块的等待与跨设备时间对齐由 ModuleGraph 的 sync 分支处理。
Tensor::Ptr Sync__::forward(const Tensor::Ptr input,
                            BatchedSequence::Ptr sequences_metadata) {
  return input;
}

}  // namespace llm_system

