'use strict';
(() => {
  const D = window.ARCH_DATA, $ = s => document.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const kinds = {child:'持有子对象', value:'内嵌 / 值副本', ref:'引用已有对象', create:'创建 / 临时产物', data:'数值 / 描述'};
  const colors = {child:'#3b7b5e',value:'#aa8439',ref:'#8a978e',create:'#8b78a7'};
  const W=260,H=112;
  const N=(type,x,y,role,id=type,title=type)=>({type,x,y,role,id,title});
  const E=(from,to,label,kind='child',route=null)=>({from,to,label,kind,route});
  const views = {
    overview:{name:'全局骨架', title:'三个入口对象，组织整个模拟器。',
      subtitle:'Model 建立设备上的模型描述，Scheduler 管理请求，Cluster 组织硬件并推动仿真。先看实线，再沿虚线理解对象之间的共享。',
      note:'<b>当前规模：</b>1 个 Cluster → 1 个 Node → 8 个 Device；每设备构建 1 个 LLM。箭头表示对象关系，不表示函数执行顺序。',
      width:970,height:830,nodes:[
        N('@main',350,20,'局部作用域 · 创建三个入口对象','main','eval/test.cpp · main()'),
        N('Model',20,230,'构图入口 · 逐设备创建 LLM'),N('Cluster',350,230,'硬件拓扑 · 驱动仿真与汇总统计'),N('Scheduler',680,230,'请求排队 · 为各 DP 组分配 batch'),
        N('LLM',20,450,'每设备 1 份 · 总计创建 8 份'),N('Node',350,450,'node[0] · 当前 1 个节点'),N('BatchedSequence',680,450,'running_queue · 当前 8 个 batch'),
        N('Executor',20,680,'Cluster 内嵌 · 所有设备共用'),N('Device',350,680,'device[0…7] · 各自持有执行图'),N('Sequence',680,680,'单条请求 · 长度、阶段与时间')],
      edges:[E('main','Model','局部对象','create'),E('main','Cluster','Create()','create'),E('main','Scheduler','Create()','create'),
        E('Model','LLM','循环创建，不是 8 项 module_list','create'),E('Cluster','Node','node'),E('Node','Device','device'),
        E('Scheduler','BatchedSequence','running_queue'),E('BatchedSequence','Sequence','sequence'),
        E('Model','Cluster','cluster','ref'),E('Cluster','Scheduler','scheduler','ref'),
        E('Cluster','Executor','executor','value',{points:[[350,286],[315,286],[315,736],[280,736]],label:[313,647]})]},
    models:{name:'模型组合树',title:'每个模块，如何组合下一层对象。',
      subtitle:'从每设备一份的 LLM 展开。子模块保存在继承的 Module.module_list 中；权重、激活和缓存描述保存在 tensor_list 中。',
      note:'<b>当前模型：</b>Embedding + 3 个 Decoder + 57 个 MoEDecoder + LmHead。每个 MoE 层在每设备上创建 32 个 routed expert 和 1 个 shared expert。重复层/专家仅展示一份构造结构；点击 ＋ 逐层展开。'},
    execution:{name:'Device 与执行图',title:'一份模型，三种不同的组织结构。',
      subtitle:'Module 是模型组合树，ModuleGraph 是执行记录，TimeStamp 是统计树节点。TopModuleGraph 把后两者组织到每个 Device 上。',
      note:'<b>最关键的共享：</b>Device.status 是值成员；TopModuleGraph.status 与 ModuleGraph.status 都是对它的引用。TimeStamp.status 则是另存的快照。',
      width:1070,height:900,nodes:[
        N('Device',390,20,'每设备一份 · 本图展开其中一台'),
        N('TopModuleGraph',40,230,'记录容器 + 游标 + 统计树'),N('StatusBoard',390,230,'设备实时累计状态','liveStatus'),N('ExecStatus',740,230,'当前算子 / high / low 暂存结果'),
        N('ModuleGraph',40,470,'一条进入 / 退出记录'),N('TimeBoard',390,470,'管理当前统计位置与根节点'),N('Module',740,470,'已有的模型模块 · 不被复制'),
        N('TimeStamp',390,710,'根节点及递归子节点'),N('StatusBoard',740,710,'时间戳内嵌的状态副本','snapshot')],
      edges:[E('Device','TopModuleGraph','top_module_graph'),E('Device','liveStatus','status','value'),E('Device','ExecStatus','3 个结果槽','value'),
        E('TopModuleGraph','ModuleGraph','module_graph · 有序 vector'),E('TopModuleGraph','TimeBoard','timeboard','value'),
        E('TopModuleGraph','liveStatus','status &','ref'),
        E('ModuleGraph','liveStatus','status &','ref',{points:[[300,526],[345,526],[345,286],[390,286]],label:[344,392]}),
        E('ModuleGraph','Module','module · 进入记录指针；退出为 null','ref',{points:[[170,470],[170,421],[870,421],[870,470]],label:[755,410]}),
        E('TimeBoard','TimeStamp','top · 值成员','value'),E('TimeStamp','snapshot','status','value'),
        E('TimeStamp','TimeStamp','leafstamp · 递归持有子时间戳','child',{points:[[390,766],[345,766],[345,848],[520,848],[520,822]],label:[440,873]})]},
    scheduler:{name:'请求与 batch',title:'请求是 Sequence，batch 是它们的容器。',
      subtitle:'调度器把等待请求放入每个 DP 组的 BatchedSequence，算子用 batch 元数据决定 token 数和专家负载。',
      note:'<b>当前配置：</b>8 设备 / 非专家 TP=1 → 8 个 DP 组；总 batch 容量 32 → 每组初始容量 4。请求由等待队列转入运行 batch，继续更新同一个 Sequence。',
      width:990,height:810,nodes:[
        N('Scheduler',350,20,'全局调度器 · 一份共享对象'),
        N('Sequence',20,260,'sequence_queue · 等待中的请求','waiting'),N('BatchedSequence',350,260,'running_queue · 每 DP 组一个'),N('SequenceInfo',680,260,'sequences_info · 真实轨迹模式'),
        N('Sequence',350,520,'sequence · 当前运行的请求','running'),N('ModelConfig',680,520,'引用 main 的原配置对象'),N('SystemConfig',20,520,'调度器按值保存的系统配置')],
      edges:[E('Scheduler','waiting','sequence_queue'),E('Scheduler','BatchedSequence','running_queue'),E('Scheduler','SequenceInfo','可选轨迹记录'),
        E('BatchedSequence','running','sequence'),
        E('waiting','running','移入 batch · 仍是同一请求','create'),
        E('Scheduler','SystemConfig','system_config','value',{points:[[350,76],[10,76],[10,490],[150,490],[150,520]],label:[130,479]}),
        E('Scheduler','ModelConfig','model_config &','ref',{points:[[610,76],[965,76],[965,490],[810,490],[810,520]],label:[820,479]})]},
    memory:{name:'张量与内存',title:'Tensor 描述形状，MemoryObject 描述地址。',
      subtitle:'设备的 MMapController 负责地址映射，DRAMInterface 把逻辑内存请求接到 Ramulator；算子性能还可以走理想估算。',
      note:'<b>当前 use_ramulator=off：</b>DRAMInterface 与其外部接口仍在设备构造时建立。图的下半部表示可选详细内存执行路径，不代表当前每个算子都会走它。',
      width:1010,height:1060,nodes:[
        N('Tensor',20,20,'权重 / 激活 / KV cache 的元数据'),N('Device',350,20,'此设备共享一套地址映射器'),
        N('MemoryObject',20,260,'基址、大小、访问块数'),N('MMapController',350,260,'分配模拟地址 · 拆分地址字段'),N('DRAMInterface',680,260,'外部时序接口 + 命令展开函数表'),
        N('DRAMRequest',20,520,'逻辑 Read / Write / GEMV 请求'),N('MemoryConfig',350,520,'cube / bank / row / col / 粒度'),N('@ramulator',680,520,'IFrontEnd + IMemorySystem','external','Ramulator · 两个接口'),
        N('PIMRequest',20,800,'详细路径产生的一组具体命令'),N('PIMCommand',350,800,'命令类型、地址、回指请求'),N('ExecStatus',680,800,'内存时序 / 命令计数结果')],
      edges:[E('Tensor','MemoryObject','memory_object'),E('Device','MMapController','mmap_controller'),E('Device','DRAMInterface','dram_interface'),
        E('MemoryObject','MMapController','映射器','ref'),E('MMapController','MemoryConfig','memory_config','value'),
        E('DRAMRequest','MemoryObject','operands_ · 复用张量地址','ref'),
        E('DRAMInterface','external','Factory 创建，raw pointer 保存','create'),
        E('DRAMRequest','PIMRequest','Read / Write / GEMV 函数展开','create'),
        E('PIMRequest','PIMCommand','command_queue','value'),
        E('external','ExecStatus','执行结果回到接口','create')]},
    catalog:{name:'全部对象索引',title:'完整类型索引，随时找到下一层。',
      subtitle:'搜索类名、成员名或中文用途。选中对象后，可沿成员关系、子模块构造关系和反向引用继续阅读。',
      note:'<b>范围：</b>'+esc(D.scope)+' 历史遗留 DataObject 也列出并注明；CacheKey、TensorVec、Ptr 等别名不重复计为对象。'}
  };
  let current='overview', selected=null, history=[], scale=1, toastTimer;
  const storageKey='llmsim-architecture-progress-v1';
  let progress={read:[],bookmark:null};
  function normalize(p){
    if(!p || !Array.isArray(p.read)) throw Error('格式不正确');
    return {read:[...new Set(p.read.filter(n=>!!D.classes[n]))],bookmark:p.bookmark&&D.classes[p.bookmark.type]&&views[p.bookmark.view]?p.bookmark:null};
  }
  try { const p=localStorage.getItem(storageKey);if(p)progress=normalize(JSON.parse(p)); }catch{}
  function toast(message){$('#toast').textContent=message;$('#toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('visible'),2600);}
  function save(){try{localStorage.setItem(storageKey,JSON.stringify(progress));}catch{toast('浏览器未允许保存，请导出阅读进度。');}updateProgress();}
  function updateProgress(){
    $('#read-count').textContent=`${progress.read.length} / ${Object.keys(D.classes).length} 个类型已读`;
    $('#resume').disabled=!progress.bookmark;$('#resume').textContent=progress.bookmark?`回到 ${progress.bookmark.type}`:'尚未标记';
    document.querySelectorAll('.read-dot').forEach(e=>e.style.display=progress.read.includes(e.dataset.type)?'':'none');
  }
  function wrap(s,max){let parts=[],line='',count=0;for(const c of s){const w=c.charCodeAt(0)>255?1:.56;if(count+w>max){parts.push(line);line='';count=0;}line+=c;count+=w;}if(line)parts.push(line);return parts;}
  function renderDiagram(){
    const v=views[current], nodes=new Map(v.nodes.map(n=>[n.id,n]));
    const markers=Object.entries(colors).map(([k,c])=>`<marker id="arrow-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="${c}"/></marker>`).join('');
    let edges=v.edges.map(e=>{
      const a=nodes.get(e.from),b=nodes.get(e.to);let pts,label;
      if(e.route){pts=e.route.points;label=e.route.label;}
      else if(a.y===b.y){const right=b.x>a.x;pts=[[right?a.x+W:a.x,a.y+H/2],[right?b.x:b.x+W,b.y+H/2]];label=[(pts[0][0]+pts[1][0])/2,pts[0][1]-9];}
      else {const down=b.y>a.y;const start=[a.x+W/2,down?a.y+H:a.y],end=[b.x+W/2,down?b.y:b.y+H];const mid=(start[1]+end[1])/2;pts=[start,[start[0],mid],[end[0],mid],end];label=[end[0],down?end[1]-14:end[1]+25];}
      const path=pts.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' '),dash=e.kind==='ref'?'6 5':e.kind==='create'?'2 5':'';
      return `<g><path class="edge" d="${path}" stroke="${colors[e.kind]}" ${dash?`stroke-dasharray="${dash}"`:''} marker-end="url(#arrow-${e.kind})"/><text class="edge-label" x="${label[0]}" y="${label[1]}" text-anchor="middle">${esc(e.label)}</text></g>`;
    }).join('');
    const cards=v.nodes.map(n=>{
      const c=D.classes[n.type],fontsize=n.title.length>26?13:n.title.length>21?14:16;
      const file=c?`${c.file.replace('src/','')}:${c.line}`:n.type==='@main'?'eval/test.cpp':'src/dram/ramulator2/ · 外部边界';
      return `<g class="node${selected===n.type?' selected':''}" data-type="${esc(n.type)}" transform="translate(${n.x},${n.y})" role="button" tabindex="0" aria-label="查看 ${esc(n.title)} 对象组成"><rect class="node-bg" width="${W}" height="${H}" rx="10"/><rect x="0" y="16" width="3" height="25" rx="1" fill="#75926b"/><text class="node-title" x="17" y="33" font-size="${fontsize}">${esc(n.title)}</text>${wrap(n.role,18).slice(0,2).map((l,i)=>`<text class="node-role" x="17" y="${56+i*17}">${esc(l)}</text>`).join('')}<text class="node-file" x="17" y="95">${esc(file)}</text><circle class="read-dot" data-type="${esc(n.type)}" cx="244" cy="17" r="4" style="display:${progress.read.includes(n.type)?'':'none'}"/></g>`;
    }).join('');
    $('#diagram').innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${esc(v.title)}" viewBox="0 0 ${v.width} ${v.height}" width="${v.width}" height="${v.height}"><title>${esc(v.title)}</title><desc>${esc(v.subtitle)}</desc><defs>${markers}</defs>${edges}${cards}</svg>`;
    fit();
  }
  function zoom(factor){scale=Math.max(.35,Math.min(2.2,factor));const v=views[current],svg=$('#diagram svg');if(!svg)return;svg.style.width=v.width*scale+'px';svg.style.height=v.height*scale+'px';$('#diagram').style.width=v.width*scale+'px';$('#diagram').style.height=v.height*scale+'px';}
  function fit(){if(views[current].nodes)zoom(Math.min(1.25,($('#workspace').clientWidth-24)/views[current].width,($('#workspace').clientHeight-20)/views[current].height));}
  function showView(id,updateHash=true){
    if(!views[id])id='overview';current=id;const v=views[id];
    $('#title').textContent=v.title;$('#subtitle').textContent=v.subtitle;$('#view-note').innerHTML=v.note;
    document.querySelectorAll('.nav-button').forEach(b=>{b.classList.toggle('active',b.dataset.view===id);b.setAttribute('aria-current',b.dataset.view===id?'page':'false');});
    $('#workspace').hidden=!v.nodes;$('#model-tree').hidden=id!=='models';$('#catalog').hidden=id!=='catalog';
    $('.tools').hidden=!v.nodes;
    if(v.nodes)renderDiagram();if(id==='models')renderTree();if(id==='catalog')renderCatalog();
    if(updateHash)window.history.replaceState(null,'','#'+id);
  }
  function sourceLink(file,line,label){return `<button class="source-link" data-source="${esc(file)}" data-line="${line}">${esc(label||`${file}:${line}`)} ↗</button>`;}
  function typeButton(type,label){return `<button class="type-button" data-type="${esc(type)}">${esc(label||type)} →</button>`;}
  function memberHTML(owner,m){return `<div class="member"><span class="badge ${m.kind}">${kinds[m.kind]}</span>${m.target&&D.classes[m.target]?typeButton(m.target):''}<div class="member-name" style="margin-top:7px">${esc(m.name)}</div><div class="type">${esc(m.type)}</div>${m.role?`<p>${esc(m.role)}</p>`:''}${sourceLink(D.classes[owner].file,m.line)}<details><summary>完整声明（含 & / *）</summary><pre>${esc(m.declaration)}</pre></details></div>`;}
  function childHTML(c){return `<div class="member"><span class="member-name">${esc(c.name)}</span>${typeButton(c.type)}<p>${esc(c.role)}</p>${c.conditions.length?`<div class="condition">条件：${esc(c.conditions.join(' 且 '))}</div>`:''}${sourceLink(c.file,c.line,'创建位置 · '+c.file+':'+c.line)}</div>`;}
  function inspect(type,push=true){
    if(type==='@main'){showSource('eval/test.cpp',16);return;}
    if(type==='@ramulator'){showSource('src/dram/dram_interface.cpp',16);return;}
    const c=D.classes[type];if(!c)return;
    if(push&&selected&&selected!==type)history.push(selected);selected=type;
    $('#inspector').hidden=false;$('#inspect-back').disabled=!history.length;
    const objectMembers=c.members.filter(m=>m.kind!=='data'),dataMembers=c.members.filter(m=>m.kind==='data');
    const active=c.children.filter(c=>!c.inactive),other=c.children.filter(c=>c.inactive);
    const parents=[];
    for(const p of Object.values(D.classes)){
      const memberNames=p.members.filter(m=>m.target===type).map(m=>m.name);
      if(memberNames.length)parents.push(`<div class="backlink">${typeButton(p.name)} <span class="type">${esc(memberNames.join(', '))}</span></div>`);
      else if(p.children.some(x=>x.type===type))parents.push(`<div class="backlink">${typeButton(p.name)} <span class="type">构造为子模块</span></div>`);
    }
    const derived=Object.values(D.classes).filter(x=>x.bases.includes(type));
    $('#inspect-body').innerHTML=`<span class="eyebrow">${esc(c.group.toUpperCase())} / 对象组成</span><h2>${esc(type)}</h2><p>${esc(c.summary)}</p>${sourceLink(c.file,c.line)}
      <div class="actions"><button id="bookmark" class="primary">${progress.bookmark?.type===type?'已标记在这里':'读到这里'}</button><button id="mark-read">${progress.read.includes(type)?'✓ 已读 · 取消':'标记已读'}</button></div>
      ${c.notes.map(s=>`<p class="note">${esc(s)}</p>`).join('')}
      ${c.bases.includes('Module')?`<p>继承 ${typeButton('Module')}，因此也具有 <code>module_list</code>、<code>tensor_list</code>、<code>device</code>。这些是基类成员，不是本类又声明一套。</p>`:''}
      ${objectMembers.length?`<h3 class="subheading">成员中的子对象与引用 · ${objectMembers.length}</h3>${objectMembers.map(m=>memberHTML(type,m)).join('')}`:''}
      ${c.children.length?`<h3 class="subheading">module_list 内创建的子模块</h3>${active.map(childHTML).join('')}${other.length?`<details class="more"><summary>其他条件下的子模块 · ${other.length}</summary>${other.map(childHTML).join('')}</details>`:''}`:''}
      ${c.tensors.length?`<details class="more"><summary>构造时创建的 Tensor · ${c.tensors.length} 个位置</summary>${c.tensors.map(childHTML).join('')}</details>`:''}
      ${!objectMembers.length&&!c.children.length?`<p class="note">${c.bases.includes('Module')?'本类没有再创建下一层 Module；张量与设备关联沿用基类成员。':'此类型主要保存数值、标记或描述数据。'}</p>`:''}
      ${dataMembers.length?`<details class="more"><summary>数值、标记与其余成员 · ${dataMembers.length}</summary>${dataMembers.map(m=>memberHTML(type,m)).join('')}</details>`:''}
      ${derived.length?`<details class="more"><summary>派生类型 · ${derived.length}</summary><p>${derived.map(x=>typeButton(x.name)).join(' ')}</p></details>`:''}
      ${parents.length?`<details class="more"><summary>哪些类型持有 / 引用 / 构造它 · ${parents.length}</summary>${parents.join('')}</details>`:''}`;
    $('#inspector').scrollTop=0;
    document.querySelectorAll('.node').forEach(n=>n.classList.toggle('selected',n.dataset.type===type));
    $('#bookmark').onclick=()=>{progress.bookmark={type,view:current};save();$('#bookmark').textContent='已标记在这里';toast('已保存阅读位置：'+type);};
    $('#mark-read').onclick=()=>{progress.read=progress.read.includes(type)?progress.read.filter(n=>n!==type):[...progress.read,type];save();$('#mark-read').textContent=progress.read.includes(type)?'✓ 已读 · 取消':'标记已读';};
  }
  function showSource(file,line){
    const source=D.sources[file];if(!source){toast('此文件不在源码快照范围内。');return;}
    $('#source-title').textContent=`${file}:${line}`;
    const vscode='vscode://file'+encodeURI(D.root+'/'+file)+':'+line;
    const local='../../'+file.split('/').map(encodeURIComponent).join('/');
    $('#source-links').innerHTML=`<a href="${esc(vscode)}">在 VS Code 打开此行 ↗</a><a href="${esc(local)}" target="_blank" rel="noopener">打开磁盘原文件 ↗</a><span>以下为生成时的源码快照</span>`;
    $('#source-code').innerHTML=source.text.split('\n').map((s,i)=>`<span class="code-line${i+1===Number(line)?' highlight':''}"><span class="line-no">${i+1}</span>${esc(s)||' '}</span>`).join('');
    $('#source').showModal();requestAnimationFrame(()=>$('#source-code .highlight')?.scrollIntoView({block:'center'}));
  }
  function treeNode(type,entry=null,depth=0){
    const c=D.classes[type],el=document.createElement('details');el.className='tree-node';
    const rootName=type==='LLM'?'每设备一份':entry?.name||type;
    el.innerHTML=`<summary><div><span class="tree-title">${esc(type)}</span><span class="tree-role">${esc(entry?.role||c.summary)}</span><div class="tree-name">${esc(rootName)}</div>${entry?.conditions.length?`<div class="condition">${esc(entry.conditions.join(' 且 '))}</div>`:''}</div><button class="tree-detail" data-type="${esc(type)}">组成 / 源码</button></summary><div class="tree-children"></div>`;
    let loaded=false;
    const fill=()=>{
      if(loaded)return;loaded=true;const container=el.querySelector('.tree-children');
      const active=c.children.filter(x=>!x.inactive),other=c.children.filter(x=>x.inactive);
      active.forEach(child=>container.append(treeNode(child.type,child,depth+1)));
      if(other.length){const details=document.createElement('details');details.className='tree-other';details.innerHTML=`<summary>其他配置分支 · ${other.length} 个构造位置</summary>`;other.forEach(child=>details.append(treeNode(child.type,child,depth+1)));container.append(details);}
      if(c.tensors.length){const details=document.createElement('details');details.className='tree-other';details.innerHTML=`<summary>tensor_list 中的张量描述 · ${c.tensors.length} 个构造位置</summary>`;for(const t of c.tensors){const row=document.createElement('div');row.className='tree-leaf';row.innerHTML=`${esc(t.name)} · ${esc(t.role)} ${typeButton('Tensor')}${t.conditions.length?`<div class="condition">${esc(t.conditions.join(' 且 '))}</div>`:''}<br>${sourceLink(t.file,t.line)}`;details.append(row);}container.append(details);}
      if(!c.children.length){const row=document.createElement('div');row.className='tree-leaf';row.innerHTML=`叶模块 · 通过继承的 ${typeButton('Module','Module.tensor_list / device')} 关联张量与设备。`;container.append(row);}
    };
    el.addEventListener('toggle',()=>{if(el.open)fill();});
    if(depth===0){fill();el.open=true;}
    return el;
  }
  function renderTree(){const root=$('#model-tree');root.innerHTML='';root.append(treeNode('LLM'));}
  function renderCatalog(){
    const q=$('#search').value.toLowerCase().trim();const list=Object.values(D.classes).filter(c=>JSON.stringify(c).toLowerCase().includes(q)).sort((a,b)=>a.group.localeCompare(b.group)||a.name.localeCompare(b.name));
    $('#catalog-count').textContent=`显示 ${list.length} / ${Object.keys(D.classes).length} 个类型`;
    $('#catalog-list').innerHTML=list.map(c=>`<button class="catalog-item" data-type="${esc(c.name)}"><small>${esc(c.group)}${c.bases.includes('Module')?' · 继承 Module':''}${c.name==='DataObject'?' · 历史遗留':''}</small><strong>${esc(c.name)}</strong><span>${esc(c.summary)}</span></button>`).join('')||'<p>没有匹配的类型。</p>';
  }
  function download(name,content,type){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),30000);}
  function exportSVG(){
    const v=views[current];if(!v.nodes)return;const copy=$('#diagram svg').cloneNode(true);copy.removeAttribute('style');
    copy.setAttribute('viewBox',`0 -115 ${v.width} ${v.height+115}`);copy.setAttribute('height',v.height+115);
    const ns='http://www.w3.org/2000/svg';
    // Embed all SVG styles so the exported diagram is independent of this page.
    const style=document.createElementNS(ns,'style');style.textContent=`text{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif}.node-bg{fill:#fffefa;stroke:#d5ddd1;stroke-width:1.2}.node-title{font-weight:650;fill:#263d30}.node-role{font-size:12px;fill:#69786e}.node-file{font-size:10px;fill:#8b978d}.edge{fill:none;stroke-width:1.5}.edge-label{font-size:10px;fill:#5a6d60;paint-order:stroke;stroke:#fdfdf9;stroke-width:5px}.read-dot{display:none}`;
    copy.prepend(style);const bg=document.createElementNS(ns,'rect');bg.setAttribute('x','0');bg.setAttribute('y','-115');bg.setAttribute('width',v.width);bg.setAttribute('height',v.height+115);bg.setAttribute('fill','#fdfdf9');copy.insertBefore(bg,style);
    const title=document.createElementNS(ns,'text');title.setAttribute('x','20');title.setAttribute('y','-72');title.setAttribute('font-size','24');title.setAttribute('fill','#24392f');title.textContent='LLMSimulator · '+v.name;copy.append(title);
    const legend=document.createElementNS(ns,'text');legend.setAttribute('x','20');legend.setAttribute('y','-36');legend.setAttribute('font-size','12');legend.setAttribute('fill','#68766e');legend.textContent='绿实线：子对象　金实线：值成员　灰虚线：引用　紫点线：创建/临时产物 · 按当前配置与源码归纳';copy.append(legend);
    download(`LLMSimulator-${current}.svg`,'<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(copy),'image/svg+xml');toast('已导出独立 SVG 图。');
  }
  $('#views').innerHTML=Object.entries(views).map(([id,v],i)=>`<button class="nav-button" data-view="${id}"><span>0${i+1}</span>${v.name}</button>`).join('');
  document.addEventListener('click',e=>{
    const source=e.target.closest('[data-source]');if(source){showSource(source.dataset.source,Number(source.dataset.line));return;}
    const type=e.target.closest('[data-type]');if(type&&type.tagName!=='circle'){e.preventDefault();inspect(type.dataset.type);return;}
    const view=e.target.closest('[data-view]');if(view){$('#inspector').hidden=true;showView(view.dataset.view);}
  });
  document.addEventListener('keydown',e=>{if((e.key==='Enter'||e.key===' ')&&e.target.matches('.node')){e.preventDefault();inspect(e.target.dataset.type);}if(e.key==='Escape'&&!$('dialog[open]'))$('#inspector').hidden=true;});
  $('#inspect-close').onclick=()=>$('#inspector').hidden=true;
  $('#inspect-back').onclick=()=>{const previous=history.pop();if(previous)inspect(previous,false);};
  $('#resume').onclick=()=>{if(progress.bookmark){showView(progress.bookmark.view);inspect(progress.bookmark.type);}};
  $('#search').oninput=renderCatalog;
  $('#zoom-in').onclick=()=>zoom(scale*1.2);$('#zoom-out').onclick=()=>zoom(scale/1.2);$('#fit').onclick=fit;
  $('#export-svg').onclick=exportSVG;
  $('#help-open').onclick=()=>$('#help').showModal();$('#help-close').onclick=()=>$('#help').close();$('#source-close').onclick=()=>$('#source').close();
  $('#export-progress').onclick=()=>download('llmsimulator-architecture-progress.json',JSON.stringify({...progress,format:'llmsim-architecture-v1'},null,2),'application/json');
  $('#import-progress').onclick=()=>$('#progress-file').click();
  $('#progress-file').onchange=async e=>{try{const f=e.target.files[0];if(!f)return;const p=JSON.parse(await f.text());if(p.format!=='llmsim-architecture-v1')throw Error();progress=normalize(p);save();if(selected&&!$('#inspector').hidden)inspect(selected,false);toast('阅读进度已导入。');}catch{toast('无法导入：请选择本架构图导出的进度文件。');}finally{e.target.value='';}};
  window.addEventListener('hashchange',()=>showView(location.hash.slice(1),false));
  window.addEventListener('resize',fit);
  $('#scope').textContent=`${Object.keys(D.classes).length} 个自有类型 · ${Object.keys(D.sources).length} 份源码快照`;
  updateProgress();showView(location.hash.slice(1)||'overview',false);
})();
