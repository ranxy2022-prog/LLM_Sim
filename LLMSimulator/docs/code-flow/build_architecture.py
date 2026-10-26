#!/usr/bin/env python3
"""Generate an offline object catalogue. Reads C++; writes documentation only.

Uses the same optional tree-sitter dependencies as build_flow.py. The diagrams
and relationship explanations are curated; member declarations and constructor
sites are extracted from the source. This is not a heap/runtime trace.
"""
import hashlib
import json
import re
from pathlib import Path
from build_flow import Index, ROOT, HERE, txt, flat, field, walk, declarator, arguments

# Relationship kinds: value = embedded/copy; child = held child (often shared_ptr);
# ref = same existing instance; create = construction; flow = transient result.
ROLES = {
    'Cluster.node': ('child', 'Node', '集群中的物理节点；当前配置 1 个。'),
    'Cluster.executor': ('value', 'Executor', '全体设备共用的算子性能估算与函数分发表。'),
    'Cluster.scheduler': ('ref', 'Scheduler', '引用 main 创建的同一个调度器。'),
    'Cluster.module_map': ('ref', 'Module', '按设备 rank 和完整模块名称索引已有模块，供跨设备依赖查找；不会复制模块。'),
    'Cluster.execution_time_cache': ('value', 'ExecStatus', 'CacheKey → 已计算的性能结果，减少重复估算。CacheKey 是 tuple 别名。'),
    'Node.device': ('child', 'Device', '本节点内的设备；当前配置每节点 8 个。'),
    'Node.cluster': ('ref', 'Cluster', '回指所在集群。'),
    'Device.top_module_graph': ('child', 'TopModuleGraph', '此设备的执行记录序列、执行游标和模块耗时统计。'),
    'Device.status': ('value', 'StatusBoard', '此设备本轮累计时间与能耗的唯一这份实时统计板。'),
    'Device.dram_interface': ('child', 'DRAMInterface', '对接内存时序模拟器；即使 use_ramulator=off，构造时仍创建。'),
    'Device.mmap_controller': ('child', 'MMapController', '为本设备的张量分配模拟地址并拆分 DRAM 地址字段。'),
    'Device.cluster': ('ref', 'Cluster', '访问集群共享的 Executor、模块索引和结果缓存。'),
    'Device.exec_status': ('value', 'ExecStatus', '暂存当前算子结果，之后由执行图取走并累计。'),
    'Device.high_exec_status': ('value', 'ExecStatus', '异构执行时高算力单元的结果槽。'),
    'Device.low_exec_status': ('value', 'ExecStatus', '异构执行时低算力单元的结果槽。'),
    'Model.cluster': ('ref', 'Cluster', '引用已有集群，为每个设备创建并登记 LLM。'),
    'Model.scheduler': ('ref', 'Scheduler', '引用已有调度器，取得构图所需的最大 batch 元数据。'),
    'Module.module_list': ('child', 'Module', '命名子模块表；LLM、Decoder、Attention、FFN 都通过它组合下一层对象。'),
    'Module.tensor_list': ('child', 'Tensor', '命名张量描述表：权重、激活、KV cache；保存形状与地址，不保存完整数值数组。'),
    'Module.device': ('ref', 'Device', '此模块对应的已有设备；调用其性能估算并登记执行记录。'),
    'Module.device_list': ('data', '', '参与并行/通信的设备编号列表，是 int 列表，不是 Device 对象。'),
    'Module.graph_execution': ('data', '', '是否在执行记录回放时调用本模块 forward；组合层通常只提供构图层级。'),
    'TopModuleGraph.module_graph': ('child', 'ModuleGraph', '有序 vector；每次进入/退出模块各追加一条记录，并非子图树。'),
    'TopModuleGraph.status': ('ref', 'StatusBoard', 'C++ 引用，绑定 Device.status；没有新建一份状态板。'),
    'TopModuleGraph.timeboard': ('value', 'TimeBoard', '内嵌的模块统计树管理器。'),
    'TopModuleGraph.device': ('ref', 'Device', '回指设备，取走算子结果、推进模拟状态。'),
    'TopModuleGraph.dram_powers': ('value', 'DramEnergy', 'GPU / LOGIC / PIM 对应的命令能耗系数。'),
    'TopModuleGraph.current_module': ('data', '', '指向 module_graph 的迭代器，表示本轮执行到了哪条记录。'),
    'ModuleGraph.module': ('ref', 'Module', '进入记录引用已有模块；退出记录为 nullptr。'),
    'ModuleGraph.status': ('ref', 'StatusBoard', '同样绑定 Device.status，和 TopModuleGraph 共享。'),
    'ModuleGraph.input': ('ref', 'Tensor', '进入记录保存输入；退出记录通过同一字段保存返回的张量。'),
    'ModuleGraph.input_vec': ('ref', 'Tensor', '多张量版本的输入/返回载荷，TensorVec 是共享指针列表别名。'),
    'ModuleGraph.output': ('ref', 'Tensor', '头文件保留字段；当前构造/执行实现未使用。退出载荷实际走 input。'),
    'ModuleGraph.output_vec': ('ref', 'Tensor', '头文件保留字段；当前构造/执行实现未使用。'),
    'ModuleGraph.dependency_tensor_list': ('ref', 'Tensor', '已有张量的依赖列表，用 ready 标记与设备时间控制跨设备同步。'),
    'TimeBoard.top': ('value', 'TimeStamp', '内嵌根节点，子时间戳从这里生长。'),
    'TimeBoard.current_stamp': ('ref', 'TimeStamp', '指向当前记录位置的游标，不是另一个独立时间戳。'),
    'TimeStamp.leafstamp': ('child', 'TimeStamp', '动态创建的子时间戳 raw pointer 列表；析构时递归删除。'),
    'TimeStamp.parent': ('ref', 'TimeStamp', '回指父时间戳，用于退出模块时返回上一层。'),
    'TimeStamp.status': ('value', 'StatusBoard', '状态快照/结算后的增量副本，与 Device 中持续变化的状态分开。'),
    'StatusBoard.tensor': ('ref', 'Tensor', '当前单张量状态载荷。'),
    'StatusBoard.tensor_vec': ('ref', 'Tensor', '当前多张量状态载荷。'),
    'Scheduler.sequence_queue': ('child', 'Sequence', '等待进入 batch 的请求队列。'),
    'Scheduler.running_queue': ('child', 'BatchedSequence', '每个 DP 组一个 batch 容器；当前 8 组，初始每组容量 4。'),
    'Scheduler.model_config': ('ref', 'ModelConfig', '引用 main 中的配置，不是 Model / Device 持有的副本。'),
    'Scheduler.sequences_info': ('child', 'SequenceInfo', '外部请求/专家轨迹的记录；synthesis 当前路径不使用真实轨迹。'),
    'BatchedSequence.sequence': ('child', 'Sequence', '本 batch 的运行请求；由等待队列转入，同一个 Sequence 继续更新。'),
    'BatchedSequence.scheduler': ('ref', 'Scheduler', '回指共享调度器，以获得全局专家 token 统计。'),
    'BatchedSequence.local_num_token_in_expert': ('data', '', '本 batch 各专家的 token 计数；不是专家模块对象。'),
    'BatchedSequence.num_token_in_expert': ('data', '', '聚合后的专家负载计数。'),
    'Tensor.memory_object': ('child', 'MemoryObject', '此张量的模拟地址范围与访问块数。'),
    'Tensor.device': ('ref', 'Device', '所属设备，用于申请模拟地址。'),
    'Tensor.module': ('ref', 'Module', '回指持有/登记这个张量的模块。'),
    'MemoryObject.mmap_controller': ('ref', 'MMapController', '复用设备的地址映射器，按访问块索引得到 DRAM 地址。'),
    'MMapController.memory_config': ('value', 'MemoryConfig', 'cube、channel、bank、row、col 和访问粒度等几何参数。'),
    'DRAMInterface.frontend': ('ref', 'Ramulator::IFrontEnd', 'Ramulator Factory 创建的外部前端；接收展开后的命令。'),
    'DRAMInterface.memory_system': ('ref', 'Ramulator::IMemorySystem', 'Ramulator Factory 创建的外部内存系统；推进 DRAM 时序。'),
    'DRAMInterface.pim_hw_config': ('value', 'PIMHWConfig', '处理器类型与带宽倍数的配置值。'),
    'DRAMInterface.exec_status': ('value', 'ExecStatus', '内存执行耗时与命令计数的返回结果。'),
    'DRAMInterface.kernel': ('data', '', 'Read / Write / GEMV 等命令展开函数表，不是一组 kernel 类实例。'),
    'DRAMInterface.dram_request_log_': ('value', 'DRAMRequest', '声明的日志容器；当前实现未写入。'),
    'DRAMInterface.pim_request_log_': ('value', 'DRAMRequest', '声明的日志容器；字段类型实际为 DRAMRequest，当前未写入。'),
    'DRAMRequest.operands_': ('ref', 'MemoryObject', '按输入/权重/输出等角色引用张量已有的内存对象。'),
    'PIMRequest.command_queue': ('value', 'PIMCommand', '按值存放的命令列表，供 Ramulator 前端消费。'),
    'PIMCommand.request': ('ref', 'PIMRequest', 'raw pointer 回指所属请求，用于回填统计。'),
}

NOTES = {
    'Model': ['这里的 Model 是构图入口。它循环创建 8 份设备级 LLM，随后做一次形状传播并记录各设备执行图。', '源码细节：各 LLM 的局部 name 都是 "LLM"。add_module_with_name 会覆盖同名键，所以 Model.module_list["LLM"] 最后只留最后设备那份；每台设备的 LLM 仍由 Cluster.module_map 和执行记录持有。不要把 Model.module_list 画成 8 项数组。'],
    'Module': ['这是组合模式的共同基类。所有模型层继承 Module，再用 module_list / tensor_list 保存各自的子对象。继承表示“是一个”，不是“再包含一个”。', 'getptr / Create / Ptr 多为 shared_ptr 工厂与别名；同一对象可以同时被模型树、执行记录、模块索引引用。图中的“子对象”不表示独占所有权。'],
    'LLM': ['当前 DeepSeek V3：每设备一个 LLM，含 Embedding、3 个 Decoder、57 个 MoEDecoder、LmHead。总层数 60。', '这是形状、依赖和性能建模，不是完整数值推理；图依据实际创建对象而非论文里的理想网络结构。'],
    'Decoder': ['当前 attention 键为 MultiLatentAttention；feedforward 键为 FeedForward3Way；两个 residual 键为 Residual。'],
    'MoEDecoder': ['注意按真实类型阅读：residual_1 / residual_2 名字虽像 Residual，构造函数实际调用的是 LayerNorm::Create。'],
    'MultiLatentAttention': ['当前 use_absorb=true：使用 BatchedColumn/RowParallelLinear、BatchedRoPE 和 AbsorbMLAParallel。非吸收路径仍列在“其他条件下的子模块”中。'],
    'ExpertFFN': ['当前 e_tp_dg=1，每设备 routed expert 数 = 256 / (8 / 1) = 32，每个专家为 FeedForward3Way；每设备另有 1 个 shared expert。', 'routed expert 的 use_dp=false，投影使用 Column/RowParallelLinear；shared expert 的 use_dp=true，投影直接使用 Linear。'],
    'FeedForward3Way': ['gate_proj / up_proj / down_proj 加 Activation 构成此对象。是否使用并行包装取决于 use_dp；是否创建 AllReduce 取决于 perform_all_reduce。这不是只由全局配置就能唯一决定的类型。'],
    'AbsorbMLAParallel': ['始终创建 AttentionSplit、AbsorbMLASum、AbsorbMLAGen、AttentionMerge。decode-only 并不等于不会创建 prefill/Sum 对象；其本轮输入可能为空。'],
    'TopModuleGraph': ['TopModuleGraph 是每设备一个执行容器；module_graph 是扁平的进入/退出记录序列。module_level 记录嵌套深度。', '构图时 Module::operator() 追加记录；运行时 current_module 顺序前进，遇到未就绪依赖暂缓。只对 graph_execution=true 的模块回放 forward。', '不要混淆三种结构：模型子模块树（Module）→ 执行记录序列（ModuleGraph）→ 模块统计树（TimeStamp）。'],
    'ModuleGraph': ['名字容易误导：一个 ModuleGraph 对象是一条进入/退出记录，不是又一个完整子图。模块与张量通过 shared_ptr 引用已有对象。'],
    'Device': ['每设备都有独立的 TopModuleGraph、StatusBoard、DRAMInterface、MMapController；Executor 在 Cluster 中共享。', 'GPU、LOGIC、PIM 是 ProcessorType 枚举及硬件成本函数分派，不是 Device 下的三个处理器类实例。'],
    'Executor': ['成员均为 std::function 向量，按处理器类型分派到 hardware/*_impl.cpp 的成本函数；不会真实启动 GPU kernel。', '算子输入由 Tensor 与 LayerInfo 描述，输出 ExecStatus；通过 Device 暂存，再累计到 StatusBoard。'],
    'Tensor': ['只描述 name、shape、精度、ready、并行标记与模拟地址。权重、activation、KV cache 都用 Tensor 表达。', 'MemoryObject 表示模拟内存布局；这条链不会分配实际 GPU 显存。'],
    'Scheduler': ['当前 8 个 DP 组 × 每组初始容量 4 = 总 batch 容量 32。running_queue 的一个元素是一个 BatchedSequence，不是一条请求。'],
    'DRAMInterface': ['当前 use_ramulator=off 仍会创建该接口及外部前端/内存系统；每次算子的 DRAM 访问主要走理想估算分支。', '外部 Ramulator 子工程在这里标出接口边界；本页类型索引覆盖项目自有 src，未展开嵌入的 Ramulator 插件体系。'],
    'DataObject': ['历史遗留对象，不在当前 dram/CMakeLists.txt 的编译源中。现行 Tensor 关联的是 MemoryObject。'],
    'Stat': ['仿真循环临时创建/汇总的统计记录，写入 CSV 或返回 vector<Stat>；不是 Cluster 的固定成员。'],
    'LayerInfo': ['算子 forward 中组装并传给 Device / Executor 的临时描述参数，不是 Tensor 的成员。'],
    'SequenceInfo': ['真实数据模式读取的轨迹描述。当前 synthesis 路径主要创建合成 Sequence。'],
}

CHILD_ROLES = {
    'Embedding': '输入嵌入层的形状与成本描述', 'LmHead': '输出词表投影层',
    'Decoder': '稠密前馈的 decoder 层', 'MoEDecoder': '采用专家前馈的 decoder 层',
    'MultiLatentAttention': '组合低秩 Q/KV 投影、RoPE、MLA 和输出投影',
    'Attention': '组合普通 QKV 投影、自注意力、输出投影和归约',
    'LayerNorm': '归一化的形状与成本描述', 'Residual': '残差操作的形状与成本描述',
    'FeedForward3Way': '门控前馈：gate / up / down 三路投影与激活',
    'FeedForward2Way': '两路投影与激活的前馈网络', 'ExpertFFN': '组织路由、专家前馈与跨设备通信',
    'ColumnParallelLinear': '按输出维度分片的 Linear 包装', 'RowParallelLinear': '按输入维度分片的 Linear 包装',
    'BatchedColumnParallelLinear': '按注意力头组织的批量列分片投影',
    'BatchedRowParallelLinear': '按注意力头组织的批量行分片投影',
    'Linear': '一个线性算子的权重/激活描述及成本请求',
    'BatchedLinear': '多个矩阵投影的批量线性算子',
    'Activation': '激活操作及输出形状描述', 'RoPE': '旋转位置编码描述', 'BatchedRoPE': '多张量版本的旋转位置编码',
    'AllReduce': '并行结果归约的通信成本', 'AllGather': '收集各设备分片', 'AllScatter': '分散张量分片',
    'MoEScatter': '把 token 分发到专家设备', 'MoEGather': '把专家输出收集回原 DP 组',
    'GateUpdate': '更新专家 token 数等路由元数据', 'Route': '按专家负载生成各专家输入张量描述',
    'Sync': '跨设备同步的组合包装', 'Sync__Set': '设置同步所需张量状态', 'Sync__': '等待依赖并对齐模拟时间',
    'AttentionSplit': '拆分 prefill / decode 输入', 'AttentionMerge': '合并 prefill / decode 输出描述',
    'SelfAttentionParallel': '本设备的普通 attention 分片组合',
    'MultiLatentAttentionParallel': '非吸收 MLA 的 prefill / decode 组合',
    'AbsorbMLAParallel': '吸收权重路径的 prefill / decode 组合',
    'SelfAttentionGen': 'decode 注意力与 KV cache 描述', 'SelfAttentionSum': 'prefill 注意力成本',
    'MultiLatentAttentionGen': '非吸收 MLA decode 与 KV cache 描述',
    'MultiLatentAttentionSum': '非吸收 MLA prefill 成本',
    'AbsorbMLAGen': '吸收 MLA decode 与压缩 KV / PE cache 描述',
    'AbsorbMLASum': '吸收 MLA prefill 成本',
    'CompressedKVRestore': '把压缩 KV 缓存恢复为非吸收路径所需的描述',
}


def conditions(n, stop, idx):
    result = []; inactive = False
    current = n
    while current and current != stop:
        parent = current.parent
        if parent and parent.type == 'if_statement':
            c = field(parent, 'condition')
            truth = current == field(parent, 'consequence')
            result.append(('' if truth else '非 ') + flat(c))
            value = idx.condition_value(c)
            if value is not None and bool(value) != truth:
                inactive = True
        current = parent
    return list(reversed(result)), inactive


def generate():
    idx = Index()
    classes = {}
    for path, tree in idx.trees.items():
        for n in walk(tree.root_node):
            if n.type not in ('class_specifier', 'struct_specifier') or not field(n, 'body'):
                continue
            name = txt(field(n, 'name'))
            if name not in idx.classes:
                continue
            before = idx.sources[path].splitlines()[:n.start_point.row]
            summary = next((s.split('@brief', 1)[1].strip() for s in reversed(before[-3:]) if '@brief' in s), '')
            members = []
            for c in field(n, 'body').named_children:
                if c.type != 'field_declaration': continue
                for d in c.children_by_field_name('declarator'):
                    if any(x.type == 'function_declarator' for x in walk(d)): continue
                    member = declarator(d)
                    role = ROLES.get(name + '.' + member)
                    typ = flat(field(c, 'type'))
                    if not role and typ in ('SystemConfig', 'ModelConfig'):
                        role = ('value', typ, '按值保存的一份配置副本。')
                    if not role and name == 'Executor':
                        role = ('data', '', '算子成本函数分发表；保存可调用函数，不含 Device / Tensor / ExecStatus 实例。')
                    pointer_or_ref = re.match(r'^[*&\s]+', flat(d))
                    display_type = typ + (' ' + pointer_or_ref.group().strip() if pointer_or_ref else '')
                    members.append(dict(name=member, type=display_type, declaration=flat(c), line=c.start_point.row+1,
                        kind=role[0] if role else 'data', target=role[1] if role else '',
                        role=role[2] if role else ''))
            classes[name] = dict(name=name, file=path, line=n.start_point.row+1, end=n.end_point.row+1,
                summary=summary, bases=idx.classes[name]['bases'], members=members,
                notes=NOTES.get(name, []), children=[], tensors=[], group=path.split('/')[1])
    # Extract actual direct child constructions from constructors, ignoring comments.
    for f in idx.functions.values():
        if f['owner'] not in classes or f['short'] != f['owner']: continue
        if f['owner'] != 'Module' and 'Module' not in classes[f['owner']]['bases']: continue
        for n in walk(field(f['node'], 'body')):
            if n.type != 'call_expression': continue
            callee = txt(field(n, 'function'))
            if not callee.endswith('::Create'): continue
            typ = callee.split('::')[0]
            if typ not in classes: continue
            args = arguments(field(n, 'arguments'))
            arg = args[0] if typ == 'Tensor' else (args[1] if len(args)>1 else None)
            if not arg or typ == 'LLM': continue # Model creation is covered explicitly in overview / notes.
            if typ != 'Tensor' and 'Module' not in classes[typ]['bases']: continue
            cond, inactive = conditions(n, f['node'], idx)
            child = dict(type=typ, name=flat(arg), file=f['file'], line=n.start_point.row+1,
                conditions=cond, inactive=inactive, role=CHILD_ROLES.get(typ, classes[typ]['summary']))
            if typ == 'Tensor':
                child['role'] = '张量用途：' + (flat(args[2]) if len(args)>2 else '')
            classes[f['owner']]['tensors' if typ == 'Tensor' else 'children'].append(child)
    sources = {path: dict(text=source, sha256=hashlib.sha256(source.encode()).hexdigest())
               for path, source in idx.sources.items()}
    sources['config.yaml'] = dict(text=(ROOT/'config.yaml').read_text(),
        sha256=hashlib.sha256((ROOT/'config.yaml').read_bytes()).hexdigest())
    for key, (_, target, _) in ROLES.items():
        owner, member = key.split('.')
        assert any(m['name'] == member for m in classes[owner]['members']), key
        assert not target or target in classes or target.startswith('Ramulator::'), target
    data = dict(classes=classes, sources=sources, config=idx.config, root=str(ROOT),
        scope='项目自有 src 中的完整 class / struct 定义；排除嵌入的 Ramulator 子工程、前向声明、枚举和类型别名。',
        baseline={p: s['sha256'] for p,s in sources.items()})
    (HERE/'architecture-data.js').write_text('window.ARCH_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',', ':'))+';\n')
    print(f'Architecture catalogue: {len(classes)} classes/structs, {sum(len(c["children"]) for c in classes.values())} module construction sites, {len(sources)} source snapshots.')


if __name__ == '__main__':
    import sys
    if '--check' in sys.argv:
        raw = (HERE/'architecture-data.js').read_text()
        data = json.loads(raw[len('window.ARCH_DATA = '):-2])
        changed = [p for p,h in data['baseline'].items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
        print('Source snapshot: '+('STALE: '+', '.join(changed) if changed else 'all hashes match'))
        raise SystemExit(bool(changed))
    generate()
