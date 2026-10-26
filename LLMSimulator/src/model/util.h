// 中文导读：模型结构与设备映射：组合算子模块、配置维度和并行组，在初始化阶段建立执行图。
#pragma once
#include <string>
#include <vector>

namespace llm_system {
/// @brief 按起始设备偏移和并行规模填充参与设备编号列表，供模型分布与通信模块使用。
void inline set_device_list(std::vector<int> &device_list, int device_offset,
                            int num_device) {
  device_list.resize(0);

  for (int device = device_offset; device < device_offset + num_device;
       device++) {
    device_list.push_back(device);
  }
}
}  // namespace llm_system