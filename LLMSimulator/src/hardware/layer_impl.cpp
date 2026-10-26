// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
#include "hardware/layer_impl.h"

#include "dram/dram_interface.h"
#include "hardware/device.h"
#include "module/tensor.h"

namespace llm_system {

/// @brief 先查执行缓存，未命中时为张量创建逻辑内存请求并调用 DRAM 仿真，返回统计并缓存；缓存键不含完整地址历史。
ExecStatus issueRamulator(Device_Ptr device, LayerType layer_type,
                          ProcessorType processor_type,
                          DRAMRequestType dram_request_type,
                          PIMOperandType pim_operand_type, Tensor_Ptr tensor) {
  CacheKey key = std::make_tuple(layer_type, processor_type, dram_request_type,
                                 tensor->getSize());
  ExecStatus exec_status;
  if (!device->checkExecutionCache(exec_status, key)) {
    DRAMRequest::Ptr dram_request = DRAMRequest::Create(dram_request_type);
    dram_request->AddOperand(tensor->getMemoryObject(), pim_operand_type);
    device->run_ramulator(dram_request);
    exec_status = device->dram_interface->getExecStatus(); 
    device->addExecutionCache(exec_status, key);
  }

  return exec_status;
};

/// @brief 通过设备理想内存路径估算命令计数并返回统计，供未启用 Ramulator 的能耗估算使用。
ExecStatus getIdealMemoryStatus(Device_Ptr device, ProcessorType processor_type,
                          DRAMRequestType dram_request_type, Tensor_Ptr tensor) {
  
  ExecStatus exec_status;
  long total_size = tensor->getSize();
  device->run_ideal(dram_request_type, tensor);
  exec_status = device->dram_interface->getExecStatus(); 
  return exec_status;
};

}  // namespace llm_system