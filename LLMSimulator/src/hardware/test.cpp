// 中文导读：硬件/执行成本建模：软件调用推进模拟状态，不表示宿主机上实际执行 GPU 内核。
// 本文件为未加入当前构建目标的独立示例，可能使用旧接口；正式入口见 eval/test.cpp。
#include <hardware/cluster.h>

using namespace llm_system;

/// @brief 独立调试/旧示例入口：演示本目录组件用法，未加入当前 run 目标；部分示例接口已过时，不能当作现行主流程。
int main() {
  SystemConfig config;
  Cluster::Ptr cluster = Cluster::Create(config, nullptr);

  return 0;
}