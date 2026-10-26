// 中文导读：公共类型与检查工具，供模型、调度器和硬件层共同使用。
#pragma once
#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <string>

// always fail
/// @brief 输出错误和当前工作目录后以失败状态退出程序；不是可恢复的异常。
inline void fail(const std::string &msg) {
  std::cerr << "ERROR: " << msg << std::endl << std::flush;
  std::cout << std::filesystem::current_path() << std::endl;
  std::exit(EXIT_FAILURE);
}

/// @brief 条件为 false 时调用 fail 终止程序；这是项目自己的检查函数，不受 NDEBUG 关闭。
inline void assertTrue(bool expr, const std::string &msg) {
  if (!expr) {
    fail(msg);
  }
}

/// @brief 条件为 true 时调用 fail 终止程序。
inline void assertFalse(bool expr, const std::string &msg) {
  assertTrue(!expr, msg);
}

// Only produce warnings

/// @brief 向日志流输出警告，不终止程序。
inline void warn(const std::string &msg) {
  std::clog << "WARN: " << msg << std::endl;
}

/// @brief 条件为 false 时输出警告，继续执行。
inline void checkTrue(bool expr, const std::string &msg) {
  if (!expr) {
    warn(msg);
  }
}

/// @brief 条件为 true 时输出警告，继续执行。
inline void checkFalse(bool expr, const std::string &msg) {
  checkTrue(!expr, msg);
}

/// @brief 输出指定功能尚未实现的警告，不提供替代实现。
inline void notYetImplemented(const std::string &name) {
  warn(name + " not yet implemented");
}
