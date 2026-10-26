LLMSimulator 代码阅读地图
=======================

打开方式
--------
直接用浏览器打开本目录的 index.html。无需安装依赖、编译或启动模拟器。
HTML、JavaScript、样式和源码快照都在本目录；使用时不访问网络。
移动或复制时，请保留本目录中的 index.html、app.js、style.css、flow-data.js。

如果浏览器禁止 file:// 页面保存进度，可以在本目录启动静态服务：

    python3 -m http.server 8765 --bind 127.0.0.1

然后打开 http://127.0.0.1:8765 。file:// 与 HTTP 的进度互相独立，
可以使用左侧的「导出进度 / 导入进度」迁移。

如何阅读
--------
1. 初始列从 eval/test.cpp 的 main() 开始，节点从上到下排列。
2. 点击「展开子流程」，在右侧查看被调用函数；父调用点保持原位。
3. 顶部调用路径和「返回调用点」可逐层返回并恢复视图位置。
4. 节点同时显示调用文件:行号、目标定义文件:行号；点击位置可查看源码。
5. 使用「读到这里」保存当前位置与调用路径，使用「标为已读」记录进度。
   同一函数的不同调用位置、同一位置的不同父调用路径，分别记录。
6. 条件框显示当前配置能确定的分支，其他条件路径默认折叠。
   运行时才能确定的条件会保留两条路径；展开后按状态选择阅读。
7. 循环只画一轮结构，不把实际预热、设备遍历或模型执行次数改成一次。
8. 多态与函数表节点可选择具体实现；不同候选实现不是顺序执行关系。
9. 在空白处拖动可平移；右上角调整缩放。按 / 搜索，Esc 关闭源码面板。
   搜索结果通过从 main 出发的调用路径打开，保留返回上下文。

当前范围
--------
基于生成时的工作区源码和 config.yaml：DeepSeek V3 / B100 / GPU /
1 节点 × 8 设备 / decode / 合成请求；Ramulator 性能模拟及 PD 分离关闭。
其他配置分支仍保留。构图期间与正式仿真期间的 forward 调用分别解释。

保留项目自有的工厂、构造、基类/成员初始化、operator()、forward、
性能函数表、状态累加运算符和显式析构等跳转。
简单取值、标准库、字符串比较、断言等不展开内部实现。
Ramulator 保留项目自有适配层，第三方实现停在边界处。
所有节点按源码而非注释中的理想行为组织；中文概要来自现有源码注释。

这是针对本项目的静态阅读图，不是一次实际运行的调用 trace。
对象状态、缓存命中、依赖就绪等条件不能仅由 YAML 确定，图中明确保留。
同一表达式中的嵌套调用先于外层调用；参数之间未规定的求值先后不作保证。
条件中的短路求值、提前返回、循环控制均保留。递归可按需逐层进入，
不会预先无限展开。页面默认恢复当前浏览器上次的调用路径与进度。

源码与配置保持只读。此目录中的流程图不加入 CMake，不参与模拟器运行。

检查与更新
----------
以下检查只读取现有源码；Python 标准库即可运行：

    python3 build_flow.py --check
    python3 check_flow.py

前者校验所有源码快照的 SHA-256 和节点位置，能发现代码更新后的行号过期；
后者还检查入口顺序、工厂/构造关系、嵌套调用、条件分支和关键虚函数分派。

更新图数据需要两个解析依赖，可安装到独立虚拟环境：

    python3 -m pip install -r requirements-build.txt
    python3 build_flow.py

生成器只写本目录的 flow-data.json 与 flow-data.js，不修改项目文件。
它以 C++ 语法树定位函数、控制结构和调用，结合本项目的类型、继承关系、
模块注册与函数表解析间接跳转。它不是通用 C++ 编译器；未来源码若新增
复杂模板、宏生成调用或新的动态注册模式，应重新审阅生成数据。
当前快照的调用接收对象均能解析；具体运行时目标仍通过候选实现表示。
改动源码后，先重新生成并通过检查，再继续使用图。旧进度按调用路径匹配，
源码移动可能使部分旧标记失效；更新前建议导出进度备份。

可选浏览器检查（需要 Playwright 及其 Chromium）：

    python3 -m pip install playwright==1.55.0
    python3 -m playwright install chromium
    python3 check_flow.py --browser

文件
----
index.html / app.js / style.css : 离线交互界面
flow-data.js                   : 浏览器直接加载的图与源码快照
flow-data.json                 : 同一份结构化数据，便于核对
build_flow.py                  : 只读源码索引与图数据生成器
check_flow.py                  : 数据校验和可选浏览器交互检查
requirements-build.txt         : 重新生成数据所需依赖；浏览页面不需要

阅读进度保存在浏览器 localStorage。清除站点数据、换浏览器或换打开地址
可能看不到原进度；导出的 JSON 可用于恢复。页面本身不会上传数据。

对象架构图
----------
打开 architecture.html，可从“对象包含关系”理解项目：
1. 全局骨架：main 创建 Model、Cluster、Scheduler；节点、设备、batch 的关系。
2. 模型组合树：从 LLM 展开 Decoder / MoEDecoder、注意力、FFN、专家、算子和张量。
   重复的层/专家只显示一份构造结构，其他配置分支默认折叠。
3. Device 与执行图：区分 Module 模型树、ModuleGraph 执行记录序列、TimeStamp 统计树。
4. 请求与 batch：Scheduler、BatchedSequence、Sequence 与可选轨迹的关系。
5. 张量与内存：Tensor、MemoryObject、映射器、DRAM 适配层与外部接口。
6. 全部对象索引：79 个自有 class / struct，含成员、继承、反向引用和源码位置。

绿色实线表示持有子对象，金色实线表示内嵌/值成员；灰色虚线引用已有对象，
紫色点线表示创建或临时产物。子对象可以通过 shared_ptr 被多个地方同时持有。
点击对象查看说明；“读到这里”和“标记已读”独立于原调用流程图保存。
右上角可缩放并导出独立 SVG。architecture-overview.svg 与
architecture-execution.svg 是已导出的总览、设备执行图，可单独查看或分享。

移动架构图时保留 architecture.html、architecture.css、architecture.js、
architecture-data.js；静态 SVG 不依赖其他文件。原调用图与架构图互相提供入口。

检查源码快照（仅需 Python 标准库）：
    python3 build_architecture.py --check

更新类型目录与源码快照（依赖同 requirements-build.txt）：
    python3 build_architecture.py

成员声明、构造位置从 C++ 语法树提取；职责说明和重点关系图经过人工整理。
配置改变后，除重新生成数据外，还需复核 architecture.js 中的图、配置数量及
build_architecture.py 中的职责说明；生成脚本不声称自动适配任意新架构。
本图不运行仿真，也不修改源码；Ramulator 内部对象在适配层的接口边界截止。
