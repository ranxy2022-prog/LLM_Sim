// 中文导读：模型模块与执行元数据：主要描述张量形状、依赖和性能成本，不进行完整的神经网络数值推理。
// 本文件为未加入当前构建目标的独立示例，可能使用旧接口；正式入口见 eval/test.cpp。
#include "module/timeboard.h"

using namespace llm_system;

/// @brief 独立调试/旧示例入口：演示本目录组件用法，未加入当前 run 目标；部分示例接口已过时，不能当作现行主流程。
int main() {
  TimeBoard timeboard;
  StatusBoard status;
  timeboard.push_timestamp(status, "level1");
  timeboard.push_timestamp(status, "level2");
  timeboard.pop_timestamp(status);
  timeboard.push_timestamp(status, "level3");

  timeboard.print();
}
