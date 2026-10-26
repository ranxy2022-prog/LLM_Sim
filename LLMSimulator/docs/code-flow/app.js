/* Offline viewer. The model and source snapshots come from flow-data.js. */
(() => {
  'use strict';
  const D = window.FLOW_DATA, F = D.functions;
  const $ = id => document.getElementById(id);
  const el = (tag, cls, content) => { const e = document.createElement(tag); if (cls) e.className = cls; if (content !== undefined) e.textContent = content; return e; };
  const button = (label, cls, fn) => { const b = el('button', cls, label); b.type = 'button'; b.addEventListener('click', fn); return b; };
  const storageKey = 'llmsimulator-code-flow-v1';
  let state = { read: {}, bookmark: null, stack: [{fid:D.entry,key:'main',dispatch:[]}], open: {}, zoom:1, scroll:{x:0,y:0} };
  let storageAvailable = true;
  try { const saved = JSON.parse(localStorage.getItem(storageKey)); if (saved && validProgress(saved)) state = {...state,...saved}; }
  catch (_) { storageAvailable = false; }
  const workspace = $('workspace');
  const escapeKey = s => s.replace(/[^a-zA-Z0-9_-]/g, '_');
  const cardId = (frame,s) => 'card-'+escapeKey(frame.key+'|'+s.id);
  const stepKey = (frame,s) => frame.key+'|'+s.id;
  let toastTimer, saveTimer;
  function toast(text) { $('toast').textContent=text; $('toast').classList.add('visible');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').classList.remove('visible'),2600); }
  function validProgress(s) {
    const dict=x=>x&&typeof x==='object'&&!Array.isArray(x)&&Object.values(x).every(v=>typeof v==='boolean');
    const stack=x=>Array.isArray(x)&&x.length>0&&x.length<=100&&x[0].fid===D.entry&&x.every((f,i)=>f&&F[f.fid]&&typeof f.key==='string'&&(!f.dispatch||Array.isArray(f.dispatch)&&f.dispatch.every(t=>typeof t==='string'))&&(!i||f.caller===x[i-1].fid&&Number.isInteger(f.fromLine)&&allSteps(F[x[i-1].fid].steps).some(n=>n.id===f.via&&n.targets?.includes(f.fid))));
    return s&&typeof s==='object'&&stack(s.stack)&&dict(s.read)&&dict(s.open)&&Number.isFinite(s.zoom)&&s.zoom>=.6&&s.zoom<=1.4&&(!s.bookmark||stack(s.bookmark.stack)&&typeof s.bookmark.step==='string'&&typeof s.bookmark.key==='string'&&s.bookmark.scroll&&Number.isFinite(s.bookmark.scroll.x)&&Number.isFinite(s.bookmark.scroll.y));
  }
  function save() { clearTimeout(saveTimer);saveTimer=setTimeout(()=>{try { localStorage.setItem(storageKey,JSON.stringify(state));storageAvailable=true; }catch(_){if(storageAvailable)toast('浏览器无法保存进度，请使用左侧「导出进度」。');storageAvailable=false;}},100); }
  function rememberView() { state.scroll={x:workspace.scrollLeft,y:workspace.scrollTop}; }
  function allSteps(steps) { const out=[]; for(const s of steps){out.push(s);for(const k of ['then','otherwise','body'])out.push(...allSteps(s[k]||[]));for(const c of s.cases||[])out.push(...allSteps(c.body));}return out; }
  function fileLink(file,line,end=line,label) { return button(label||`${file}:${line}`,'file-link',()=>showSource(file,line,end)); }
  function showSource(file,line,end) {
    $('source-panel').hidden=false;$('source-title').textContent=`${file}:${line}`;
    $('source-meta').textContent='生成时的源码快照 · 高亮为当前节点对应位置';
    const code=$('source-code');code.replaceChildren();
    const lines=D.sources[file].split('\n');
    for(let i=0;i<lines.length;i++) { const row=el('div','source-row'+(i+1>=line&&i+1<=end?' highlight':'')+(/^\s*\/\//.test(lines[i])?' comment':''));row.dataset.line=i+1;row.append(el('span','line-number',i+1),el('span','line-content',lines[i]||' '));code.append(row); }
    requestAnimationFrame(()=>{const target=code.querySelector(`[data-line="${Math.max(1,line-4)}"]`);code.scrollTop=target?target.offsetTop-code.offsetTop:0;});
  }
  $('source-close').onclick=()=>$('source-panel').hidden=true;
  function details(key,cls,summary,initial) {
    const box=el('details',cls);box.open=key in state.open?!!state.open[key]:initial;
    const title=el('summary');if(typeof summary==='string')title.textContent=summary;else title.append(summary);box.append(title);
    box.addEventListener('toggle',()=>{state.open[key]=box.open;save();requestAnimationFrame(drawConnectors);});return box;
  }
  function chooseTargets(s,frame) {
    let targets=s.targets.map(i=>F[i]);
    if(s.code.startsWith('forward(') && frame.dispatch?.length) {
      const matched=targets.filter(t=>frame.dispatch.includes(t.owner));
      if(matched.length)targets=matched.sort((a,b)=>frame.dispatch.indexOf(a.owner)-frame.dispatch.indexOf(b.owner));
    }
    // Only presentation priority: non-GPU choices remain available.
    if(s.hint?.includes('函数表分派'))targets.sort((a,b)=>Number(b.name.endsWith('GPU'))-Number(a.name.endsWith('GPU')));
    return targets;
  }
  function openCall(depth,s,target) {
    rememberView();const parent=state.stack[depth];parent.savedScroll={...state.scroll};parent.selected=s.id;
    const key=parent.key+'/'+s.id+'>'+target;
    state.stack=state.stack.slice(0,depth+1);
    state.stack.push({fid:target,key,via:s.id,caller:parent.fid,dispatch:s.dispatch||[],fromLine:s.line});
    render();save();
    requestAnimationFrame(()=>{const col=$('columns').children[depth+1];workspace.scrollLeft=Math.max(0,col.offsetLeft*state.zoom-65);workspace.scrollTop=0;rememberView();save();});
  }
  function returnTo(depth) {
    if(depth<0)return;const frame=state.stack[depth],selected=frame.selected;
    state.stack=state.stack.slice(0,depth+1);render();
    requestAnimationFrame(()=>{const saved=frame.savedScroll;if(saved){workspace.scrollLeft=saved.x;workspace.scrollTop=saved.y;}
      const step=allSteps(F[frame.fid].steps).find(s=>s.id===selected);if(step){const card=$(cardId(frame,step));card?.classList.add('flash');}
      rememberView();save();});
  }
  function markRead(frame,s,card,b) {const k=stepKey(frame,s);if(state.read[k])delete state.read[k];else state.read[k]=true;card.classList.toggle('read',!!state.read[k]);b.classList.toggle('active',!!state.read[k]);b.textContent=state.read[k]?'✓ 已读':'标为已读';updateProgress();save();}
  function bookmark(frame,s,depth) {
    rememberView();state.bookmark={key:stepKey(frame,s),step:s.id,fid:frame.fid,stack:JSON.parse(JSON.stringify(state.stack.slice(0,depth+1))),scroll:{...state.scroll},label:F[frame.fid].name+' · '+F[frame.fid].file+':'+s.line};
    document.querySelectorAll('.call-card.bookmarked').forEach(c=>c.classList.remove('bookmarked'));
    document.querySelectorAll('.bookmark-button.active').forEach(b=>{b.classList.remove('active');b.textContent='读到这里';});
    const card=$(cardId(frame,s));card?.classList.add('bookmarked');const b=card?.querySelector('.bookmark-button');if(b){b.classList.add('active');b.textContent='◆ 读到这里';}
    updateProgress();save();toast('已保存当前位置和调用路径');
  }
  function renderCall(s,frame,depth) {
    const k=stepKey(frame,s),book=state.bookmark?.key===k;
    const targets=chooseTargets(s,frame), first=targets[0];
    const card=el('article','call-card'+(state.read[k]?' read':'')+(book?' bookmarked':'')+(frame.selected===s.id?' selected':''));card.id=cardId(frame,s);card.dataset.step=s.id;
    const top=el('div','call-top');top.append(el('div','call-kicker',`${String(s.line).padStart(3,'0')} / ${targets.length>1?'DISPATCH · 分派':'CALL · 调用'}`));
    top.append(el('div','call-name',targets.length===1?first.name:s.code.split('(')[0]+'(…)'));
    top.append(el('p','call-purpose',targets.length===1?(first.summary||'进入项目定义的函数。'):(s.hint||'按调用时的参数或对象类型选择实现。')));card.append(top);
    const loc=el('div','locations');const from=el('div','location-row');from.append(el('span','location-label','从'),fileLink(F[frame.fid].file,s.line,s.end));loc.append(from);
    if(first){const to=el('div','location-row');to.append(el('span','location-label','到'),fileLink(first.file,first.line,first.line+2),...(targets.length>1?[el('span','badge',`另 ${targets.length-1} 个实现`)]:[]));loc.append(to);}card.append(loc);
    if(s.dispatch?.length)card.append(el('div','context-note',(s.dispatch.length>1?'可能的对象类型（当前配置优先）：':'对象类型：')+s.dispatch.join(' / ')+'；在下一层追踪具体 forward。'));
    if(s.hint && targets.length===1)card.append(el('div','context-note',s.hint));
    const actions=el('div','call-actions');
    actions.append(button('展开子流程 →','expand-call',()=>openCall(depth,s,first.id)));
    const read=button(state.read[k]?'✓ 已读':'标为已读','mark-read'+(state.read[k]?' active':''),()=>markRead(frame,s,card,read));
    actions.append(read,button(book?'◆ 读到这里':'读到这里','bookmark-button'+(book?' active':''),()=>bookmark(frame,s,depth)));card.append(actions);
    if(targets.length>1){const opts=details(k+':targets','targets',`选择具体实现 · ${targets.length} 个候选`,false);
      for(const target of targets){const row=el('div','target-choice');const b=button(target.name,'',()=>openCall(depth,s,target.id));
        if(target.name.endsWith('GPU')&&s.hint?.includes('函数表'))b.append(el('span','badge','当前配置'));
        row.append(b,button(`${target.file.split('/').pop()}:${target.line}`,'source-target',()=>showSource(target.file,target.line,target.line+2)));opts.append(row);}card.append(opts);}
    return card;
  }
  function renderSteps(steps,frame,depth,parent) {
    for(const s of steps){const wrap=el('div','step-wrap');const key=stepKey(frame,s);
      if(s.kind==='call')wrap.append(renderCall(s,frame,depth));
      else if(s.kind==='condition'){
        const box=el('div','condition-box'),head=el('div','condition-head');head.append(el('span','','◇ 条件判断'),el('span','condition-value',s.value===null?'运行时判断':s.value?'当前配置：成立':'当前配置：不成立'),el('code','',s.code),fileLink(F[frame.fid].file,s.line));box.append(head);
        for(const [label,children,truth] of [['成立',s.then,true],['不成立',s.otherwise,false]]){
          const other=s.value!==null&&Boolean(s.value)!==truth;
          const branch=details(key+':'+truth,'branch'+(other?' other':''),`${label}${other?' · 其他条件路径':s.value!==null?' · 当前主线':' · 需看运行时状态'}${children.length?'':' · 继续本函数'}`,s.value!==null&&Boolean(s.value)===truth);
          const content=el('div','branch-content');if(children.length)renderSteps(children,frame,depth,content);else content.append(el('div','empty-branch','该路径没有需要展开的项目调用，继续后续步骤。'));
          branch.append(content);box.append(branch);
        }
        box.append(el('div','merge','↓ 未 return / break / continue 的路径，在此继续'));wrap.append(box);
      }else if(s.kind==='loop'){
        const header=el('span');header.append(el('span','','↻ 循环 · 展示一轮结构'),el('code','',s.code));
        const box=details(key+':loop','loop-box',header,true);box.querySelector('summary').className='loop-head';
        const body=el('div','loop-body');renderSteps(s.body,frame,depth,body);box.append(body,el('div','loop-back','↶ 满足条件时重复；循环结束后继续下方步骤'));wrap.append(box);
      }else if(s.kind==='alternative'){
        const box=details(key+':alternative','condition-box branch other',s.code,false),body=el('div','branch-content');renderSteps(s.body,frame,depth,body);box.append(body);wrap.append(box);
      }else if(s.kind==='switch'){
        const box=el('div','condition-box');box.append(el('div','switch-head','◇ switch '+s.code));
        for(const c of s.cases){const branch=details(key+':'+c.label,'branch',c.label,false);const body=el('div','branch-content');renderSteps(c.body,frame,depth,body);branch.append(body);box.append(branch);}wrap.append(box);
      }else if(s.kind==='return'||s.kind==='control'){
        const terminal=el('div','terminal',s.code);terminal.append(el('small','',s.kind==='return'?'↩ 退出本函数，回到调用点':s.code.startsWith('break')?'↳ 离开最近的循环或 switch':s.code.startsWith('continue')?'↶ 进入下一轮循环':'↗ 转入异常处理路径'));wrap.append(terminal);
      }else{const note=el('div','note-card',s.code);note.append(el('br'),fileLink(F[frame.fid].file,s.line,s.end));wrap.append(note);}
      parent.append(wrap);
    }
  }
  function render() {
    $('columns').replaceChildren();$('breadcrumbs').replaceChildren();
    state.stack.forEach((frame,depth)=>{
      const f=F[frame.fid],col=el('section','flow-column');col.dataset.depth=depth;
      const head=el('header','column-heading'),level=el('div','level');level.append(el('span','',`${String(depth).padStart(2,'0')} / ${depth?'子流程':'程序入口'}`));
      if(depth)level.append(button('↩ 返回', '',()=>returnTo(depth-1)));
      head.append(level,el('h2','',f.name),fileLink(f.file,f.line,f.end));
      if(depth){const incoming=el('p');incoming.append(el('span','','调用来自 '),fileLink(F[frame.caller].file,frame.fromLine));head.append(incoming);}
      head.append(el('p','',f.summary||'按源码顺序阅读本函数中的关键跳转。'));col.append(head);
      const body=el('div','flow-body');if(D.notes[f.name])body.append(el('div','stage-note',D.notes[f.name]));
      if(f.name==='main')body.append(el('div','stage-note','先读取 YAML 并设置配置，再创建下方对象。字符串比较、配置字段赋值和文件名拼接已省略；每个调用点保留原始源码位置。'));
      renderSteps(f.steps,frame,depth,body);
      const end=el('div','flow-end',depth?'本层结束 · 回到调用点后继续':'main 结束');if(depth){end.append(el('br'),button('↩ 回到 '+F[state.stack[depth-1].fid].name,'',()=>returnTo(depth-1)));}body.append(end);col.append(body);$('columns').append(col);
      if(depth)$('breadcrumbs').append(el('span','','›'));
      $('breadcrumbs').append(button(f.name,'',()=>returnTo(depth)));
    });
    const empty=el('div','empty-column');empty.append(el('div','empty-icon','↳'),el('h3','','下一层，从一个调用开始。'),el('p','','点击左侧节点的「展开子流程」，这里会出现被调用函数的执行流程。调用点会留在原处，方便随时返回。'));$('columns').append(empty);
    $('back').disabled=state.stack.length<=1;setZoom(state.zoom,false);updateProgress();requestAnimationFrame(drawConnectors);
  }
  function drawConnectors() {
    const svg=$('connectors');svg.replaceChildren();const ns='http://www.w3.org/2000/svg';
    const origin=$('canvas').getBoundingClientRect(),z=state.zoom;
    for(let depth=1;depth<state.stack.length;depth++){
      const frame=state.stack[depth],parent=state.stack[depth-1],step=allSteps(F[parent.fid].steps).find(s=>s.id===frame.via);if(!step)continue;
      const from=$(cardId(parent,step)),to=$('columns').children[depth]?.querySelector('.column-heading');if(!from||!to||!from.getClientRects().length)continue;
      const a=from.getBoundingClientRect(),b=to.getBoundingClientRect();const x1=(a.right-origin.left)/z,y1=(a.top+a.height/2-origin.top)/z,x2=(b.left-origin.left)/z,y2=(b.top+32-origin.top)/z;
      const path=document.createElementNS(ns,'path');path.setAttribute('d',`M${x1},${y1} C${x1+32},${y1} ${x2-35},${y2} ${x2-7},${y2}`);path.setAttribute('class','connector-line');svg.append(path);
      const arrow=document.createElementNS(ns,'path');arrow.setAttribute('d',`M${x2-9},${y2-4} L${x2-2},${y2} L${x2-9},${y2+4}`);arrow.setAttribute('fill','none');arrow.setAttribute('stroke','#82a39a');svg.append(arrow);
      const label=document.createElementNS(ns,'text');label.setAttribute('x',x2-49);label.setAttribute('y',y2-10);label.setAttribute('class','connector-label');label.textContent='进入';svg.append(label);
    }
  }
  function setZoom(value,persist=true) {state.zoom=Math.max(.6,Math.min(1.4,Number(value)||1));$('canvas').style.zoom=state.zoom;$('zoom-reset').textContent=Math.round(state.zoom*100)+'%';if(persist)save();requestAnimationFrame(drawConnectors);}
  function updateProgress() {$('read-count').textContent=`${Object.keys(state.read).length} 个调用点已读`;$('resume').disabled=!state.bookmark;$('bookmark-label').textContent=state.bookmark?.label||'在节点上点书签，保存当前位置。';}
  function revealStep(frame,id) {
    function visit(steps){for(const s of steps){if(s.id===id)return true;for(const k of ['then','otherwise','body'])if(visit(s[k]||[])){const key=stepKey(frame,s);state.open[key+(s.kind==='loop'?':loop':s.kind==='alternative'?':alternative':k==='then'?':true':':false')]=true;return true;}for(const c of s.cases||[])if(visit(c.body)){state.open[stepKey(frame,s)+':'+c.label]=true;return true;}}return false;}visit(F[frame.fid].steps);
  }
  $('resume').onclick=()=>{const b=state.bookmark;if(!b||!b.stack.every(s=>F[s.fid]))return;state.stack=JSON.parse(JSON.stringify(b.stack));const frame=state.stack.at(-1);revealStep(frame,b.step);render();requestAnimationFrame(()=>{workspace.scrollLeft=b.scroll.x;workspace.scrollTop=b.scroll.y;const s=allSteps(F[frame.fid].steps).find(s=>s.id===b.step);if(s){const card=$(cardId(frame,s));card?.scrollIntoView({block:'center',inline:'center'});card?.classList.add('flash');}rememberView();save();});};
  $('back').onclick=()=>returnTo(state.stack.length-2);
  $('zoom-in').onclick=()=>setZoom(state.zoom+.1);$('zoom-out').onclick=()=>setZoom(state.zoom-.1);$('zoom-reset').onclick=()=>setZoom(1);
  function resetHome(){state.stack=[{fid:D.entry,key:'main',dispatch:[]}];render();workspace.scrollLeft=0;workspace.scrollTop=0;rememberView();save();}
  $('home').onclick=e=>{e.preventDefault();resetHome();};
  // Search navigates via an actual call path from main, preserving return context.
  const routes=new Map([[D.entry,[]]]),visited=new Set([D.entry+'|']);const queue=[{fid:D.entry,dispatch:[],path:[]}];
  for(let q=0;q<queue.length;q++){
    const current=queue[q],fid=current.fid;
    for(const s of allSteps(F[fid].steps))if(s.targets){
      for(const target of chooseTargets(s,current).map(t=>t.id)){
        const path=[...current.path,{fid,step:s.id,target}],dispatch=s.dispatch||[],key=target+'|'+dispatch.join(',');
        if(!routes.has(target))routes.set(target,path);
        if(!visited.has(key)){visited.add(key);queue.push({fid:target,dispatch,path});}
      }
    }
  }
  function goFunction(fid) {if(!routes.has(fid)){toast('该函数没有从 main 可达的调用路径。');return;}state.stack=[{fid:D.entry,key:'main',dispatch:[]}];for(const edge of routes.get(fid)){const parent=state.stack.at(-1),s=allSteps(F[parent.fid].steps).find(x=>x.id===edge.step);parent.selected=s.id;revealStep(parent,s.id);state.stack.push({fid:edge.target,key:parent.key+'/'+s.id+'>'+edge.target,via:s.id,caller:parent.fid,dispatch:s.dispatch||[],fromLine:s.line});}render();requestAnimationFrame(()=>{const col=$('columns').children[state.stack.length-1];workspace.scrollLeft=col.offsetLeft*state.zoom;workspace.scrollTop=0;rememberView();save();});}
  const landmarks=[['从 main 开始','main'],['创建调度器','Scheduler::Create'],['创建集群与设备','Cluster::Create'],['创建模型与执行图','Model::Model'],['调度与仿真循环','Cluster::runIteration'],['请求与 batch','Scheduler::initRunningQueue'],['设备与算子执行','Cluster::run'],['统计与输出','Cluster::setTimeBreakDown']];
  landmarks.forEach(([title,name],i)=>{const fid=Object.keys(F).find(k=>F[k].name===name);const b=button('','',()=>goFunction(fid));b.append(el('span','landmark-number',String(i+1).padStart(2,'0')),el('span','',title));$('landmarks').append(b);});
  $('search').addEventListener('input',()=>{const q=$('search').value.trim().toLowerCase();$('search-results').replaceChildren();if(!q)return;
    const found=Object.values(F).filter(f=>routes.has(f.id)&&(f.name+' '+f.file).toLowerCase().includes(q)).slice(0,35);
    for(const f of found){const b=button(f.name,'',()=>{goFunction(f.id);$('search-results').replaceChildren();$('search').value='';});b.append(el('small','',f.file+':'+f.line));$('search-results').append(b);}if(!found.length)$('search-results').append(el('small','','没有匹配的可达函数。'));
  });
  const cfg=D.config;$('config-summary').append(el('strong','',cfg['model.model_name']),el('div','',`${cfg['system.gpu_gen']} · ${cfg['system.num_node']} 节点 × ${cfg['system.num_device']} 设备`));
  const pills=el('div');for(const tag of [cfg['system.processor_type'],cfg['system.optimization.decode_mode']?'decode':'prefill / decode','合成请求'])pills.append(el('span','config-pill',tag));$('config-summary').append(pills,el('div','','其他条件分支保留，默认折叠。'));
  $('map-stats').textContent=`${routes.size} 个可达函数 · ${Object.keys(D.sources).length} 份源码快照`;
  function openHelp(){$('help').showModal();}for(const id of ['help-open','help-top'])$(id).onclick=openHelp;$('help-close').onclick=()=>$('help').close();
  $('help').addEventListener('click',e=>{if(e.target===$('help')){const r=$('help').getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$('help').close();}});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){$('source-panel').hidden=true;$('search-results').replaceChildren();}if(e.key==='/'&&!['INPUT','TEXTAREA'].includes(document.activeElement.tagName)){e.preventDefault();$('search').focus();}});
  $('export-progress').onclick=()=>{rememberView();const blob=new Blob([JSON.stringify({...state,format:'llmsimulator-code-flow-v1'},null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='llmsimulator-reading-progress.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);toast('阅读进度已导出');};
  $('import-progress').onclick=()=>$('progress-file').click();$('progress-file').addEventListener('change',async()=>{const file=$('progress-file').files[0];if(!file)return;try{if(file.size>5000000)throw Error();const imported=JSON.parse(await file.text());if(!validProgress(imported))throw Error();state={...state,...imported};render();save();toast('阅读进度已恢复');}catch(_){toast('无法导入：文件格式不正确，或对应的源码版本已变化。');}$('progress-file').value='';});
  let drag=null;
  workspace.addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest('button,a,input,summary,.call-card,.source-row,code'))return;drag={x:e.clientX,y:e.clientY,sx:workspace.scrollLeft,sy:workspace.scrollTop};workspace.setPointerCapture(e.pointerId);workspace.classList.add('dragging');});
  workspace.addEventListener('pointermove',e=>{if(!drag)return;workspace.scrollLeft=drag.sx+drag.x-e.clientX;workspace.scrollTop=drag.sy+drag.y-e.clientY;});
  const endDrag=()=>{drag=null;workspace.classList.remove('dragging');};workspace.addEventListener('pointerup',endDrag);workspace.addEventListener('pointercancel',endDrag);
  workspace.addEventListener('scroll',()=>{rememberView();save();});window.addEventListener('resize',drawConnectors);
  window.addEventListener('pagehide',()=>{try{localStorage.setItem(storageKey,JSON.stringify(state));}catch(_){}});
  const savedScroll=state.scroll;render();requestAnimationFrame(()=>{workspace.scrollLeft=savedScroll?.x||0;workspace.scrollTop=savedScroll?.y||0;});
})();
