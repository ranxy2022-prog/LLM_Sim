// 中文导读：请求生命周期与 batch 调度：sum 指 prefill，gen 指 decode；记录模拟时间，不等待真实墙钟时间。
#pragma once
#include <chrono>  // for timing of loading sequence
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common/type.h"

namespace llm_system {

/// @brief 轨迹中的请求描述：保存输入/输出长度以及各 token、各 MoE 层的专家选择列表。
struct SequenceInfo {
  using Ptr = std::shared_ptr<SequenceInfo>;
  int Lin;
  int Lout;
  int cur;
  std::vector<std::vector<int>> expert_list;
};

/// @brief 单个推理请求的状态：记录输入/输出长度、已处理长度、当前轮 token 数及到达/完成时间。
class Sequence {
 public:
  using Ptr = std::shared_ptr<Sequence>;

  inline static int count_id = 0;

  /// @brief Sequence 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(int expert_seq_id = 0, int input_len = 256,
                                  int output_len = 256) {
    return Ptr(new Sequence(expert_seq_id, input_len, output_len));
  }

  // Sequence operator=(Sequence seq) {
  //   Sequence temp;
  //   temp.input_len = seq.input_len;
  //   temp.max_len = seq.max_len;
  //   temp.num_process_token = seq.num_process_token;
  //   temp.current_len = seq.current_len;
  //   return temp;
  // }

  int id;
  int expert_seq_id;
  int input_len;
  int output_len;
  int total_len;
  // 本轮获准处理的 token 数；更新请求进度后清零，下一轮由 Scheduler 再次分配。
  int num_process_token;
  // 已处理到的序列长度；与 input_len 比较决定属于 prefill 还是 decode。
  int current_len;

  int num_sum_iter;
  time_ns gen_start_time;
  time_ns arrival_time;
  time_ns first_token_time;
  time_ns end_token_time;
  time_ns queueing_delay;

  /// @brief 把 num_process_token 累加到 current_len，更新阶段/时延记录并清零本轮 token 数，等待下一轮分配。
  void update(time_ns time);
  /// @brief 设置请求允许开始 decode 的模拟时间，供分离模式的调度条件判断。
  void setGenStartTime(time_ns time);

  bool record;
  bool sum_stage;
  bool get_expert_from_list;

 private:
  /// @brief 分配请求编号并初始化长度与时间状态；total_len 按 input_len + output_len - 1 计算，初始本轮 token 数为 0。
  Sequence(int expert_seq_id, int input_len, int output_len);
  std::vector<std::vector<int>> expert_list;  // 2차원 벡터 type,sequence안의
                                              // token 수만큼 expert들이 존재.
};

// TODO: count_id 초기화를 scheduler에서하든 해야함
// int Sequence::count_id = 0;

/// @brief Scheduler 的前向声明：此处只声明类型，完整结构见对应头文件。
class Scheduler;
/// @brief 一个 DP 组的请求集合：分类 prefill/decode 请求，汇总 token 数、上下文长度和专家负载。
class BatchedSequence {
 public:
  // static int cur;
  using Ptr = std::shared_ptr<BatchedSequence>;
  using Scheduler_ptr = std::shared_ptr<Scheduler>;

  /// @brief BatchedSequence 的静态工厂：调用构造函数并建立 shared_ptr 管理，返回指向同一对象的共享指针，调用方无需手工 delete。
  [[nodiscard]] static Ptr Create(int num_expert = 8, int top_k = 2,
                                  Scheduler_ptr scheduler = nullptr) {
    return Ptr(new BatchedSequence(num_expert, top_k, scheduler));
  }

  /// @brief 保存专家参数和调度器引用，创建零值专家计数数组并从第 0 个专家层开始。
  BatchedSequence(int num_expert, int top_k, Scheduler_ptr scheduler)
      : num_expert(num_expert), top_k(top_k), scheduler(scheduler) {
    for (int i = 0; i < num_expert; i++) {  // initialize
      local_num_token_in_expert.push_back(0);
      num_token_in_expert.push_back(0);
    }
    cur_layer = 0;
  }

  /// @brief 把请求的共享指针追加到本 batch 的 sequence 数组；不复制请求，也不在此修改阶段状态。
  void add(Sequence::Ptr seq);

  /// @brief 返回本 batch 全部请求指针，调用方可继续访问同一请求对象。
  std::vector<Sequence::Ptr> get_seq();
  /// @brief 筛选 current_len 小于 input_len 的请求，即仍有输入上下文待处理的 prefill 请求。
  std::vector<Sequence::Ptr> get_sum();
  /// @brief 筛选 current_len 已达到 input_len 的请求，作为 decode 阶段的请求集合。
  std::vector<Sequence::Ptr> get_gen();
  /// @brief 对本轮实际处理的 token，按轨迹或均匀/偏斜采样统计专家负载，并推进当前专家层索引；不执行真实 gating 网络。
  void update_expert(int num_expert, int top_k, bool need_new_expert);
  /// @brief 把当前专家 token 计数归零，为下一次路由统计做准备。
  void clear_expert();
  /// @brief 遍历 batch 中的请求并调用 Sequence::update，推进本轮处理进度。
  void update(time_ns time);
  /// @brief 按请求 id 移除本 batch 中匹配的请求，为后续请求释放 batch 槽位。
  void pop(Sequence::Ptr seq);
  /// @brief 返回 batch 中请求数量，不是本轮 token 数。
  int get_num_seq();
  /// @brief 求和本 batch 所有请求的 num_process_token，得到本轮总工作量。
  int get_process_token();
  /// @brief 求和 decode 请求本轮处理 token 数。
  int get_gen_process_token();
  /// @brief 求和 prefill 请求本轮处理 token 数。
  int get_sum_process_token();
  /// @brief 计算 decode 请求的平均当前上下文长度；无 decode 请求时返回 0。
  int get_average_sequence_length();
  /// @brief 累加 decode 请求的 current_len，用于历史 KV 访问量和 MLA 恢复形状估算。
  int get_total_sequence_length();

  /// @brief 创建指定数量、输入长度和输出长度的合成请求，直接加入 batch。
  void add_dummy_sequence(int num_seq, int input_len, int output_len);

  /// @brief 按轨迹编号查询输入/输出长度，创建请求并加入 batch。
  void add_sequence(std::vector<int> seq_ids);

  int cur_layer;  // 이 batch의 현재 layer와 stage
  int num_expert;
  int top_k;
  std::vector<int> local_num_token_in_expert;
  std::vector<int> num_token_in_expert;
  std::vector<Sequence::Ptr> sequence;
  std::vector<int> seq_ids;
  Scheduler_ptr scheduler;
  // batch 通过它查询专家轨迹/采样函数；Scheduler 又持有 batch，注意此处形成共享指针环。
};

}  // namespace llm_system
