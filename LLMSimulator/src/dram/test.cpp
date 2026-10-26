// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
// 本文件为未加入当前构建目标的独立示例，可能使用旧接口；正式入口见 eval/test.cpp。
#include "dram/dram_interface.h"
#include "dram/memory_object.h"
#include "hardware/device.h"
#include "module/tensor.h"

using namespace llm_system;

/// @brief 独立调试/旧示例入口：演示本目录组件用法，未加入当前 run 目标；部分示例接口已过时，不能当作现行主流程。
int main() {
  // SystemConfig config;
  SystemConfig config = B100;
  Device::Ptr device = Device::Create(config, 0, nullptr);
  // Tensor::Ptr tensor = Tensor::Create("dummy", {7168, 1536}, "act", device, device->model_config.precision_byte);

  int mat_row = 7168;
  int mat_col = 1536;
  int precision_byte = 1;
  hw_metric memory_bandwidth = 8000; // GB/s

  // Tensor::Ptr tensor1 = Tensor::Create("dummy1", {8, 7168}, "act", device, precision_byte);
  Tensor::Ptr tensor2 = Tensor::Create("dummy2", {7168, 1536}, "weight", device, precision_byte);
  // device->setMemoryObject(tensor1);
  
  // DRAMInterface::Ptr interface =
  //     DRAMInterface::Create("./dram_config.yaml", 0.41667);
  
  DRAMInterface::Ptr interface =
      DRAMInterface::Create("./dram_config_HBM3E_192GB.yaml", 0.5);      

  interface->setPIMHWConfig(ProcessorType::GPU, 0);


  // DRAMRequest::Ptr dram_request1 = DRAMRequest::Create(DRAMRequestType::kRead);
  // dram_request1->AddOperand(tensor1->getMemoryObject(), PIMOperandType::kDRAM);

  DRAMRequest::Ptr dram_request2 = DRAMRequest::Create(DRAMRequestType::kRead);
  dram_request2->AddOperand(tensor2->getMemoryObject(), PIMOperandType::kDRAM);

  double start_time = interface->time;
  std::list<DRAMRequest::Ptr> request1;
  std::list<DRAMRequest::Ptr> request2;
  // request1.push_back(dram_request1);
  request2.push_back(dram_request2);
  // interface->HandleRequest(request1, 0);
  time_ns time1 = interface->time;
  interface->HandleRequest(request2, 0);
  time_ns time2 = interface->time;
  std::cout << "Ramulator converted Value: " << interface->time << " (ns)"<<  std::endl;
  std::cout << "Ramulator Bandwidth: " << mat_row * mat_col * precision_byte / interface->time << " GB/s" << std::endl;

  std::cout << "Estimated Value: " << mat_row * mat_col * precision_byte / memory_bandwidth << " (ns)" << std::endl;
  std::cout << "Estimated Bandwidth: " << memory_bandwidth << " GB/s" << std::endl;
  // interface->HandleRequest(request, 0);
  // std::cout << interface->time << std::endl;

  // auto interface2 =
  //  DRAMInterface::Create("./dram_config.yaml", power, 1.5);
  return 0;
}