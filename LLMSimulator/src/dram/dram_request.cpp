// 中文导读：自有 DRAM 适配层：把张量地址描述转换为内存请求，连接 Ramulator 或理想估算路径。
#include "dram/dram_request.h"

#include "common/assert.h"

namespace llm_system {

// Duration of DRAMRequests are not determined at the moment of construction
/// @brief 保存逻辑内存请求类型；此时尚未生成具体命令，也没有请求耗时结果。
DRAMRequest::DRAMRequest(DRAMRequestType type) : type_(type) {}

}  // namespace llm_system