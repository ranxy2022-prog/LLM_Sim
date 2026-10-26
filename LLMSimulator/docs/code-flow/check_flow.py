#!/usr/bin/env python3
"""Graph regression checks; optionally exercise the offline page in Chromium."""
import argparse
import json
import re
from pathlib import Path

from build_flow import HERE, iterate_steps, validate


def check_graph(data):
    validate(data)
    functions=data['functions']
    def function(name):
        return next(f for f in functions.values() if f['name']==name)
    def calls(name):
        return [s for s in iterate_steps(function(name)['steps']) if s['kind']=='call']
    def names(s):return [functions[t]['name'] for t in s['targets']]
    main=calls('main')
    required=['Scheduler::Create','Cluster::Create','Model::Model','Cluster::checkMemorySize',
              'Cluster::set_dependency','Scheduler::getActualArrivalTime','Cluster::runIteration']
    actual=[names(s)[0] for s in main if any(n in required for n in names(s))]
    assert actual==required,actual
    assert next(s for s in main if 'Model::Model' in names(s))['line']==233
    assert [names(s)[0] for s in calls('Scheduler::Create')]==['Scheduler::Scheduler','Scheduler::initRunningQueue']
    assert [names(s)[0] for s in calls('Cluster::Cluster')]==['Executor::Executor']
    distribute=calls('Model::model_distribute')
    assert [names(s)[0] for s in distribute]==['Scheduler::getMaxMetadata','Cluster::get_device','Tensor::Create','Module::operator()']
    assert distribute[-1]['dispatch']==['LLM']
    for owner,expected in [('Decoder','Residual'),('MoEDecoder','LayerNorm')]:
        residual=[s for s in calls(owner+'::forward') if s['code'].startswith('(*residual_1)')]
        assert residual and all(s['dispatch']==[expected] for s in residual)
    assert next(s for s in calls('AbsorbMLAParallel::forward') if s['code'].startswith('(*attention_gen)'))['dispatch']==['AbsorbMLAGen']
    assert names(calls('Executor::executeLinear')[0])==['LinearExecutionGPU','LinearExecutionLogic','LinearExecutionPIM']
    assert names(calls('DRAMInterface::GeneratePIMCommand')[0])==['PIM_KERNEL::Read_kernel','PIM_KERNEL::Write_kernel','PIM_KERNEL::GEMV']
    graph_calls=[s for s in calls('ModuleGraph::run') if 'forward' in s['code']]
    assert graph_calls and all('LLM::forward' not in names(s) for s in graph_calls)
    assert all('SiluAndMul::forward' not in names(s) for s in graph_calls)
    assert any('ExecStatus::operator+=' in names(s) for s in calls('LinearExecutionGPU'))
    assert 'ModelConfig::ModelConfig' in names(calls('Device::Device')[0])
    assert 'TimeBoard::TimeBoard' in names(calls('TopModuleGraph::TopModuleGraph')[0])
    assert any('TimeStamp::~TimeStamp' in names(s) for s in calls('TimeStamp::deleteTimeStamp'))
    tensor_vec=next(f for f in functions.values() if f['name']=='Module::operator()' and 'TensorVec' in f['signature'])
    pop=next(s for s in tensor_vec['steps'] if s['kind']=='call' and 'pop_module_graph' in s['code'])
    assert 'TensorVec' in functions[pop['targets'][0]]['signature']
    split=next(s for s in iterate_steps(function('Cluster::runIteration')['steps']) if s['kind']=='condition' and 'disagg_system' in s['code'])
    assert split['value'] is False
    assert any('Cluster::runIterationMixed' in names(s) for s in iterate_steps(split['otherwise']) if s['kind']=='call')
    for f in functions.values():
        ids=[s['id'] for s in iterate_steps(f['steps'])]
        assert len(ids)==len(set(ids)),('Duplicate step ID',f['name'])
    assert not data['unresolved'],data['unresolved']
    assert data['constants']['system_config.use_inject_rate'] is False
    assert data['constants']['system_config.request_per_second']==10
    print('PASS: entry order, constructors, nested calls, conditional branches, dynamic dispatch and source locations.')


def check_browser():
    from playwright.sync_api import sync_playwright, expect
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(viewport={'width':1500,'height':1000},accept_downloads=True)
        errors=[];requests=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:requests.append(r.url))
        page.goto((HERE/'index.html').as_uri())
        expect(page.locator('.flow-column')).to_have_count(1)
        def column():return page.locator('.flow-column').last
        def card(name,col=None):
            return (col or column()).locator('.call-card').filter(has=page.locator('.call-name',has_text=re.compile('^'+re.escape(name)+'$'))).first
        def enter(name):
            c=card(name);c.locator('.expand-call').click();page.wait_for_timeout(120)
        # Current configuration keeps optional exports collapsed.
        expect(card('Cluster::exportGantt')).not_to_be_visible()
        expect(card('TopModuleGraph::print_timeboard')).to_be_visible()
        # Enter, mark and resume at a specific invocation, not globally per function.
        enter('Scheduler::Create')
        expect(column().locator('h2')).to_have_text('Scheduler::Create')
        enter('Scheduler::Scheduler')
        expect(column().locator('h2')).to_have_text('Scheduler::Scheduler')
        c=card('Scheduler::initExpertList');c.locator('.mark-read').click();c.locator('.bookmark-button').click()
        expect(page.locator('#read-count')).to_have_text('1 个调用点已读')
        c.locator('.file-link').first.click()
        expect(page.locator('#source-panel')).to_be_visible()
        expect(page.locator('#source-code .highlight')).to_have_count(1)
        expect(page.locator('#source-title')).to_have_text('src/scheduler/scheduler.cpp:25')
        page.locator('#source-close').click()
        page.wait_for_timeout(200);page.reload();page.wait_for_timeout(250)
        expect(page.locator('.flow-column')).to_have_count(3)
        expect(card('Scheduler::initExpertList')).to_have_class(re.compile('read'))
        page.locator('#home').click();page.locator('#resume').click()
        expect(column().locator('h2')).to_have_text('Scheduler::Scheduler')
        expect(card('Scheduler::initExpertList')).to_have_class(re.compile('bookmarked'))
        page.locator('#back').click()
        expect(column().locator('h2')).to_have_text('Scheduler::Create')
        expect(card('Scheduler::Scheduler')).to_have_class(re.compile('selected'))
        # Factory initRunningQueue calls the correct inline constructor, including scoped Ptr aliases.
        enter('Scheduler::initRunningQueue');enter('BatchedSequence::Create');enter('BatchedSequence::BatchedSequence')
        expect(column().locator('h2')).to_have_text('BatchedSequence::BatchedSequence')
        expect(column().locator('.loop-box')).to_have_count(0)  # only standard vector operations
        # Source snapshot, breadcrumb navigation, real call route search and dynamic target narrowing.
        page.locator('#search').fill('Model::model_distribute')
        page.locator('#search-results button').first.click()
        enter('Module::operator()')
        expect(card('LLM::forward')).to_be_visible()
        expect(column().locator('.call-name',has_text='Attention::forward')).to_have_count(0)
        enter('LLM::forward')
        expect(column().locator('h2')).to_have_text('LLM::forward')
        expect(column().locator('.branch.other').first).not_to_have_attribute('open','')
        page.locator('#zoom-out').click();expect(page.locator('#zoom-reset')).to_have_text('90%')
        page.locator('#zoom-reset').click()
        page.screenshot(path='/tmp/llmsim-code-flow-detail.png',full_page=True)
        # Progress backup contains the invocation path and can be restored.
        with page.expect_download() as download:page.locator('#export-progress').click()
        saved=Path(download.value.path()).read_text();progress=json.loads(saved)
        assert len(progress['read'])==1 and progress['bookmark']['stack'][-1]['fromLine']==30
        page.locator('#progress-file').set_input_files({'name':'progress.json','mimeType':'application/json','buffer':saved.encode()})
        expect(page.locator('#toast')).to_have_text('阅读进度已恢复')
        page.locator('#progress-file').set_input_files({'name':'invalid.json','mimeType':'application/json','buffer':b'{"stack": []}'})
        expect(page.locator('#toast')).to_contain_text('无法导入')
        page.locator('#help-top').click();expect(page.locator('#help')).to_be_visible();page.keyboard.press('Escape')
        # Representative large and alternate flows must render without JavaScript errors.
        for name in ['Cluster::runIteration','MultiLatentAttention::forward','ExpertFFN::forward','ModuleGraph::run','Executor::executePType','LinearExecutionGPU','DRAMInterface::GeneratePIMCommand','Cluster::setTimeBreakDown']:
            page.locator('#search').fill(name)
            page.locator('#search-results button').first.click()
            expect(column().locator('h2')).to_have_text(name)
        # Same callee / same nested call, reached from different parent call sites.
        page.locator('#home').click();enter('Cluster::Create')
        init=column().locator('.call-card').filter(has=page.locator('.call-name',has_text=re.compile('^Cluster::initializeDRAM$')))
        expect(init).to_have_count(3)
        init.nth(0).locator('.expand-call').click()
        card('Device::initializeDRAM').locator('.mark-read').click()
        page.locator('#back').click()
        column().locator('.call-card').filter(has=page.locator('.call-name',has_text=re.compile('^Cluster::initializeDRAM$'))).nth(1).locator('.expand-call').click()
        expect(card('Device::initializeDRAM')).not_to_have_class(re.compile(r'\bread\b'))
        page.set_viewport_size({'width':800,'height':760});page.locator('#home').click()
        expect(page.locator('#home')).to_be_visible();expect(page.locator('.flow-column')).to_have_count(1)
        assert not errors,errors
        assert all(url.startswith('file:') for url in requests),requests
        browser.close()
    print('PASS: offline load, nested navigation, return, source preview, bookmarks, reload, search, zoom, export/import, and alternate flows.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser',action='store_true')
    args=parser.parse_args()
    data=json.loads((HERE/'flow-data.json').read_text())
    check_graph(data)
    if args.browser:check_browser()
