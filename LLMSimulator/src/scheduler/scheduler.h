// 中文导读：请求生命周期与 batch 调度：sum 指 prefill，gen 指 decode；记录模拟时间，不等待真实墙钟时间。
#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>


#include "hardware/hardware_config.h"
#include "model/model_config.h"
#include "scheduler/sequence.h"

namespace llm_system {

/// @brief 请求调度器：维护等待请求与各 DP 组的运行 batch，分配每轮 token 预算并推进请求状态。
class Scheduler : public std::enable_shared_from_this<Scheduler> {
 public:
  using Ptr = std::shared_ptr<Scheduler>;
  // Ptr 是类型别名，不是另一个调度器对象。复制它会共享同一个 Scheduler。

  /// @brief 创建共享调度器后初始化各 DP batch；initRunningQueue 会用 shared_from_this，必须在 shared_ptr 建立后调用。
  [[nodiscard]] static Ptr Create(SystemConfig system_config,
                                  ModelConfig& model_config,
                                  std::string expert_file_path = "",
                                  int batch_size = 4096,
                                  int num_max_batched_token = 4096 * 64,
                                  int max_process_token = 8192 * 4) {
    Ptr ptr = Ptr(new Scheduler(system_config, model_config, expert_file_path,
                                batch_size, num_max_batched_token,
                                max_process_token));
    ptr->initRunningQueue();
    return ptr;
  };

  /// @brief 通过 shared_from_this 返回当前 Scheduler 的共享指针；对象须已由 shared_ptr 管理，不能在普通构造期间调用。
  Ptr getPtr() { return shared_from_this(); }

  /// @brief 清空等待队列和运行 batch 容器；重新使用前需调用 initRunningQueue 建立 batch。
  void clear();

  /// @brief 按 real_data 选择合成请求或轨迹请求，加入 sequence_queue；合成长度超上限时会调整输出长度。
  void pushSeq(int num_seq);
  /// @brief 创建固定输入/输出长度的合成请求，设置 KV 复用或 prefill/decode 初态，再加入等待队列；长度随机扰动代码当前被注释。
  void pushDummySeq(int input_len = 256, int max_len = 1024);
  /// @brief 从已加载轨迹中抽取请求长度并加入等待队列；当前实现跳过输出长度为 512 的记录。
  void pushRealSeq(int num_seq);

  /// @brief 为各 DP batch 直接添加指定长度的合成请求，供模型构建或示例初始化使用。
  void initializeDummyInput(int num_seq, int input_len, int output_len);

  /// @brief 为各 DP batch 抽取轨迹请求编号并直接加入，区别于先进入等待队列的 pushSeq。
  void initializeRealInput(int num_seq);

  /// @brief 反复分配 token、推进请求并补队列，以形成稳定初态；不调用硬件模型，也不累计真实算子耗时。
  void hittingQueue(int iter);

  /// @brief 检查任一运行 batch 是否还有 prefill 请求，供阶段选择与分离模式记账使用。
  bool hasSumSeq();

  /// @brief 返回所有 DP 组的 batch 智能指针列表；复制指针列表，不深拷贝请求状态。
  std::vector<BatchedSequence::Ptr> getAllMetadata();
  /// @brief 按 DP rank 返回对应的运行 batch，用于设备取得本组请求元数据。
  BatchedSequence::Ptr getMetadata(int dp_rank);
  /// @brief 构造初始化模型所需的 batch 描述，用预设 token 预算和 batch 容量建立占位请求。
  BatchedSequence::Ptr getMaxMetadata(int num_expert, int top_k, Ptr scheduler = nullptr);

  /// @brief 为运行请求分配本轮 token 数并返回各 DP batch；非分离模式有 prefill 时优先处理 prefill，decode 通常每请求一步一个 token。
  std::vector<BatchedSequence::Ptr> setMetadata();
  /// @brief 按本轮耗时更新各 batch 的请求状态，移除达到 total_len 的请求，并返回首次生成/结束事件供统计。
  std::vector<Sequence::Ptr> updateScheduler(time_ns time = 0);
  /// @brief 分离模式下只推进 prefill 请求，设置可开始 decode 的时刻并收集首次生成/完成事件。
  std::vector<Sequence::Ptr> updateSchedulerSumGenSplit(time_ns time = 0);

  /// @brief 打印首个 DP batch 的 token 数、prefill/decode 数量及平均上下文长度，便于观察负载。
  void printStatus();

  // random
  /// @brief 在专家编号范围内均匀采样，去重后返回 top_k 个专家；要求 top_k 不超过专家数。
  std::set<int> getRandomExpert(int top_k);
  /// @brief 按给定权重离散采样并去重，返回 top_k 个专家；权重分布由函数内静态对象保存。
  std::set<int> getZipfianRandomExpert(std::vector<double> skewness_weight, int top_k);
  /// @brief 从 token_id 开始轮转选取不同专家，使专家编号分配尽量均匀。
  std::set<int> getEquallyDistributedExpert(int token_id, int top_k);
  /// @brief 用固定初始种子的随机引擎抽取轨迹编号，再对总请求数取模；要求轨迹非空。
  int getRandomExpertSeqId();
  /// @brief 生成均值 0、标准差 0.4 的正态样本，供合成请求长度扰动逻辑使用。
  static double getNormaldistribution();
  /// @brief 以请求速率参数为均值生成泊松整数样本，使用固定初始种子的静态引擎。
  static int getPoissondistribution(int request_per_second);
  /// @brief 旧的随机注入辅助函数：随机决定是否注入以及注入数量，不是当前 fillSequenceQueue 的到达表路径。
  static int getNumInjection();

  /// @brief 预生成 num_iter*100 个到达间隔并累加为纳秒时间表；当前对间隔使用泊松分布，并非标准泊松过程的指数间隔，重复调用会追加。
  void getActualArrivalTime(int num_iter);

  /// @brief 启用注入率时统计 [total_time-iter_time, total_time) 内到达的请求；否则按 batch 空位补充等待请求。默认零时间用于初始化。
  void fillSequenceQueue(time_ns iter_time = 0, time_ns total_time = 0);
  // 默认参数写在声明处：无参调用等价于 fillSequenceQueue(0, 0)。

  // time to fill running queue
  /// @brief 从等待队列头部取请求，轮转填入未满的 DP batch；启用到达率时记录排队延迟，time 参数当前未使用。
  void fillRunningQueue(time_ns time = 0);

  /// @brief 累计所有 DP batch 的请求数；区别于 running_queue.size() 所表示的 DP 组数。
  int getBatchSize();
  /// @brief 累计本轮 num_process_token 非零的 prefill 请求数，而不是所有尚未完成 prefill 的请求数。
  int getSumSize();
  /// @brief 累计本轮 num_process_token 非零的 decode 请求数，而不是所有等待执行的 decode 请求数。
  int getGenSize();

  /// @brief 累加各 batch 的 decode 上下文总长度后除以总 batch 请求数；空 batch 返回 0。
  int getAverageSeqlen();
  /// @brief 累计所有 DP batch 本轮分配的 token 数，供模型形状、空载判断和统计使用。
  int getNumProcessToken();

  int total_seq_num;

  time_ns total_time;

  int total_batch_size;
  int batch_size_per_dp;  // per dp
  int num_max_batched_token;
  int max_process_token;
  int dp_degree;

  bool real_data;
  bool real_expert_data;

  std::vector<Sequence::Ptr> sequence_queue;
  // 等待入场的请求在 sequence_queue；运行中的请求按 DP 组保存在下面各个 batch 内。
  // running_queue.size() 是 DP 组数；实际运行请求数由 getBatchSize() 汇总。
  std::vector<BatchedSequence::Ptr> running_queue;
  
  std::vector<time_ns> actual_arrival_time; 
  // 指向下一个待分配给合成请求的到达时刻；它不是 GPU 执行图的位置。
  int cur_arrival_time_idx;
  std::vector<int> total_token_in_expert;

  SystemConfig system_config;
  ModelConfig& model_config;
  std::vector<SequenceInfo::Ptr> sequences_info;
  std::string expert_file_path;
  
  /// @brief 清空旧运行 batch，并为每个 DP 组创建一个 BatchedSequence；此时各 batch 尚无请求。
  void initRunningQueue(); // juhwan
  
 private:
  /// @brief 保存配置并计算 DP 数及每组 batch 容量，初始化专家轨迹；运行 batch 由 Create 随后建立。
  Scheduler(SystemConfig system_config, ModelConfig& model_config,
            std::string expert_file, int batch_size, int num_max_batched_token,
            int max_process_token);

  /// @brief 加载专家路由 CSV 并建立请求长度/专家轨迹；synthesis 路径不读取文件，部分失败路径退回合成负载。
  void initExpertList(std::string expert_file_path);
  // void initRunningQueue();

  bool disagg_system = false;
};

}  // namespace llm_system
