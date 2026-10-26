#!/usr/bin/env python3
"""Read-only C++ source indexer for this project's offline reading map.

Install tree-sitter==0.25.2 and tree-sitter-cpp==0.23.4 to regenerate.
Only files beside this script are written. No compiler or simulator is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def txt(n):
    return n.text.decode('utf-8') if n else ''


def flat(n):
    return re.sub(r'\s+', ' ', txt(n)).strip()


def field(n, name):
    return n.child_by_field_name(name) if n else None


def arguments(n):
    """Comments are named AST children but never function arguments."""
    return [c for c in n.named_children if c.type!='comment'] if n else []


def walk(n):
    if n:
        yield n
        for c in n.named_children:
            yield from walk(c)


def declarator(n):
    while n and n.type not in ('identifier', 'field_identifier', 'qualified_identifier', 'operator_name', 'destructor_name'):
        n = field(n, 'declarator') or (n.named_children[0] if n.named_children else None)
    return txt(n)


def scalar(v):
    v = v.strip()
    if v in ('true', 'on'): return True
    if v in ('false', 'off'): return False
    if re.fullmatch(r'-?\d+(\.\d+)?', v): return float(v) if '.' in v else int(v)
    return v.strip('"\'')


def yaml_values():
    result, stack = {}, []
    for line in (ROOT / 'config.yaml').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        indent = len(line) - len(line.lstrip())
        key, value = line.strip().split(':', 1)
        while stack and stack[-1][0] >= indent: stack.pop()
        value = value.split('#', 1)[0].strip()
        if value: result['.'.join([p[1] for p in stack] + [key])] = scalar(value)
        else: stack.append((indent, key))
    return result


class Index:
    def __init__(self):
        from tree_sitter import Language, Parser
        import tree_sitter_cpp
        parser = Parser(Language(tree_sitter_cpp.language()))
        self.sources, self.trees, self.functions = {}, {}, {}
        self.classes, self.symbols = {}, defaultdict(list)
        self.unknown = []
        files = [ROOT / 'eval/test.cpp'] + sorted(p for p in (ROOT / 'src').rglob('*')
            if p.suffix in ('.cpp', '.h') and 'ramulator2' not in p.parts and p.name != 'test.cpp')
        for path in files:
            name = path.relative_to(ROOT).as_posix()
            source = path.read_bytes()
            self.sources[name] = source.decode('utf-8')
            self.trees[name] = parser.parse(source)
        for path, tree in self.trees.items(): self.catalog(tree.root_node, path)
        for f in self.functions.values():
            self.symbols[f['name']].append(f['id'])
        self.config = yaml_values()
        self.constants = self.make_constants()
        self.active_owner = ''
        self.member_aliases = self.collect_member_aliases()
        self.module_types = self.collect_module_types()
        self.constructed_module_types={t for types in self.module_types.values() for t in types}
        self.dispatch_tables = self.collect_dispatch_tables()
        self.executable_types = self.collect_executable_types()

    def catalog(self, n, path, cls='', namespace=''):
        if n.type == 'namespace_definition':
            name = txt(field(n, 'name'))
            if name != 'llm_system': namespace = '::'.join(filter(None, [namespace, name]))
        if n.type in ('class_specifier', 'struct_specifier'):
            cls = txt(field(n, 'name'))
            if not field(n, 'body'): return
            base = next((c for c in n.named_children if c.type == 'base_class_clause'), None)
            bases = [txt(c) for c in base.named_children if c.type in ('type_identifier', 'qualified_identifier')] if base else []
            members = {};locations={}
            for c in field(n, 'body').named_children:
                if c.type != 'field_declaration': continue
                typ = txt(field(c, 'type'))
                for d in c.children_by_field_name('declarator'):
                    if any(x.type == 'function_declarator' for x in walk(d)): continue
                    members[declarator(d)] = typ + ('*' if '*' in txt(d) else '')
                    locations[declarator(d)]=(path,c.start_point.row+1)
            self.classes[cls] = {'bases': bases, 'members': members, 'locations':locations}
        if n.type == 'function_definition':
            d = field(n, 'declarator')
            fd = next((x for x in walk(d) if x.type == 'function_declarator'), None)
            if not fd or not field(n, 'body'): return
            name = declarator(field(fd, 'declarator'))
            if '::' in name:
                owner = name.rsplit('::', 1)[0]
            else:
                owner = cls
                name = '::'.join(filter(None, [cls or namespace, name]))
            params = [p for p in field(fd, 'parameters').named_children if 'parameter_declaration' in p.type]
            param_types = [txt(field(p, 'type')) for p in params]
            identity = name + '(' + ','.join(param_types) + ')'
            fid = hashlib.sha1(identity.encode()).hexdigest()[:12]
            before = self.sources[path].splitlines()[:n.start_point.row]
            summary = ''
            for line in reversed(before[-4:]):
                if '@brief' in line:
                    summary = line.split('@brief', 1)[1].strip(); break
            body = flat(field(n, 'body'))
            trivial = bool(re.fullmatch(r'\{ return (?:shared_from_this\(\)|[\w.>\-]+|\w+\.(?:at\(\w+\)|size\(\)));? \}', body))
            self.functions[fid] = dict(id=fid, name=name, owner=owner, short=name.rsplit('::', 1)[-1],
                file=path, line=n.start_point.row+1, end=n.end_point.row+1,
                summary=summary, node=n, fd=fd, params=params, param_types=param_types,
                min_args=sum(field(p, 'default_value') is None for p in params),
                result=txt(field(n, 'type')), trivial=trivial, signature=flat(d))
            return
        for c in n.named_children: self.catalog(c, path, cls, namespace)

    def members(self, cls):
        data = self.classes.get(cls, {})
        result = {}
        for base in data.get('bases', []): result.update(self.members(base))
        result.update(data.get('members', {}))
        return result

    def cls_of(self, typ):
        if typ == 'T': return 'Module'
        candidates = [w for w in re.findall(r'\b[A-Za-z_]\w*\b', typ) if w in self.classes]
        if candidates: return candidates[-1]
        for name in self.classes:
            if re.search(r'\b' + name + r'_[Pp]tr\b', typ): return name
        if typ == 'TensorVec': return 'Tensor'
        return ''

    def lookup(self, name, argc=None):
        options = self.symbols.get(name, [])
        if not options and '::' in name:
            owner, short = name.rsplit('::', 1)
            for base in self.classes.get(owner, {}).get('bases', []):
                options += self.lookup(base + '::' + short, argc)
        if argc is not None:
            matching = [i for i in options if self.functions[i]['min_args'] <= argc <= len(self.functions[i]['params'])]
            if matching: options = matching
        return list(dict.fromkeys(options))

    def make_constants(self):
        c = {}
        for k, v in self.config.items():
            tail = k.rsplit('.', 1)[-1]
            if k.startswith('system.'):
                c['system_config.'+tail] = v
                c['config.'+tail] = v
                c['device.config.'+tail] = v
            if tail in ('model_name','compressed_kv','use_absorb','skewness','precision_byte'):
                c['model_config.'+tail] = v
        c['model_name'] = self.config.get('model.model_name')
        c['processor_type_name'] = self.config.get('system.processor_type')
        c['data_name'] = self.config.get('simulation.data')
        for name in ('exit_out_of_memory','mem_cap_limit'):
            for prefix in ('system_config.','config.','device.config.'):
                c[prefix+name]=self.config.get('simulation.'+name)
        # Read the selected ModelConfig's actual constructor arguments, including defaults.
        constructors = self.lookup('ModelConfig::ModelConfig')
        if constructors:
            params = self.functions[constructors[0]]['params']
            for node in walk(self.trees['src/model/model_config.h'].root_node):
                if node.type == 'init_declarator' and declarator(field(node, 'declarator')) == c['model_name']:
                    value = field(node, 'value')
                    args = field(value, 'arguments')
                    if args:
                        for i, param in enumerate(params):
                            vals=arguments(args)
                            val = vals[i] if i < len(vals) else field(param, 'default_value')
                            key = 'model_config.'+declarator(field(param, 'declarator'))
                            if val and key not in c: c[key] = scalar(txt(val))
        # The selected hardware preset supplies flags not assigned by main (including
        # use_inject_rate); do not infer them from similarly named YAML entries.
        hardware=self.lookup('SystemConfig::SystemConfig')
        if hardware:
            params=self.functions[hardware[0]]['params']
            for node in walk(self.trees['src/hardware/hardware_config.h'].root_node):
                if node.type=='init_declarator' and declarator(field(node,'declarator'))==self.config.get('system.gpu_gen'):
                    args=field(field(node,'value'),'arguments')
                    if args:
                        for i,param in enumerate(params):
                            vals=arguments(args)
                            val=vals[i] if i<len(vals) else field(param,'default_value')
                            name=declarator(field(param,'declarator'))
                            if val and val.type in ('true','false','number_literal','string_literal'):
                                for prefix in ('system_config.','config.','device.config.'):
                                    if prefix+name not in c:c[prefix+name]=scalar(txt(val))
        return c

    def collect_member_aliases(self):
        aliases=defaultdict(dict)
        for f in self.functions.values():
            if f['short']!=f['owner']:continue
            members=self.members(f['owner'])
            for n in walk(field(f['node'],'body')):
                if n.type!='assignment_expression':continue
                left,right=txt(field(n,'left')),flat(field(n,'right')).replace('->','.')
                if left in members and right in self.constants:aliases[f['owner']][left]=self.constants[right]
        if self.config.get('simulation.data')=='synthesis':
            aliases['Scheduler'].update(real_data=False,real_expert_data=False,expert_file_path='none')
        return dict(aliases)

    def collect_module_types(self):
        result = defaultdict(dict)
        # Each project's factory takes parent path followed by the module's local name.
        for f in self.functions.values():
            if f['short']!=f['owner']:continue
            for n in walk(field(f['node'],'body')):
                if n.type != 'call_expression': continue
                callee = txt(field(n, 'function'))
                if not callee.endswith('::Create'): continue
                cls = callee.split('::')[0]
                if not self.lookup(cls+'::forward'): continue
                args = arguments(field(n, 'arguments'))
                if len(args)>1:
                    literal=next((v for v in walk(args[1]) if v.type=='string_literal'),None)
                    if literal:
                        priority=0;current=n
                        while current and current!=f['node']:
                            parent=current.parent
                            if parent and parent.type=='if_statement':
                                value=self.condition_value(field(parent,'condition'))
                                if value is not None and bool(value)!=(current==field(parent,'consequence')):priority=1
                            current=parent
                        key=f['owner']+'::'+scalar(txt(literal))
                        result[key][cls]=min(priority,result[key].get(cls,1))
        return {k: sorted(v,key=lambda c:(v[c],c)) for k,v in result.items()}

    def collect_executable_types(self):
        result=set()
        for f in self.functions.values():
            if f['short']!=f['owner']:continue
            for n in f['node'].named_children:
                if n.type!='field_initializer_list':continue
                for init in n.named_children:
                    if txt(init.named_children[0])!='Module':continue
                    args=next((x for x in init.named_children if x.type=='argument_list'),None)
                    vals=arguments(args)
                    if len(vals)>4 and txt(vals[4])=='true':result.add(f['owner'])
        return result

    def collect_dispatch_tables(self):
        tables = defaultdict(dict)
        for f in self.functions.values():
            if not f['name'].startswith('Executor::init'): continue
            for n in walk(field(f['node'], 'body')):
                if n.type != 'assignment_expression': continue
                left, right = txt(field(n,'left')), txt(field(n,'right'))
                m = re.match(r'(\w+_function_ramulator_ptr)\.at\(\(int\)ProcessorType::(\w+)\)', left)
                if m and self.lookup(right): tables[m[1]][m[2]] = self.lookup(right)
        return dict(tables)

    def env_for(self, f):
        env = self.members(f['owner'])
        env['this'] = f['owner']
        for p in f['params']: env[declarator(field(p,'declarator'))] = txt(field(p,'type'))
        for n in walk(field(f['node'], 'body')):
            if n.type not in ('declaration','for_range_loop'): continue
            typ = txt(field(n,'type'))
            for d in n.children_by_field_name('declarator'):
                key = declarator(d)
                if not key or '(' in key: continue
                if typ == 'auto':
                    val = field(d, 'value') or field(n,'right')
                    inferred = self.infer(val, env, f)
                    if inferred: env[key] = inferred
                else: env[key] = f['owner']+'::Ptr' if typ=='Ptr' else typ
        return env

    def environment_at(self, f, node):
        """Respect block scopes, including same-named iterators in separate loops."""
        env=self.members(f['owner']);env['this']=f['owner']
        for p in f['params']:
            typ=txt(field(p,'type'))
            env[declarator(field(p,'declarator'))]=f['owner']+'::Ptr' if typ=='Ptr' else typ
        chain=[];current=node
        while current and current!=f['node']:
            chain.append(current);current=current.parent
        for current in reversed(chain):
            parent=current.parent
            if parent and parent.type=='compound_statement':
                for sibling in parent.named_children:
                    if sibling.start_byte>=current.start_byte:break
                    if sibling.type=='declaration': self.bind(sibling,env,f)
            if current.type in ('for_range_loop','for_statement'):
                self.bind(current,env,f)
                init=field(current,'initializer')
                if init and init.type=='declaration':self.bind(init,env,f)
        return env

    def bind(self,n,env,f):
        typ=txt(field(n,'type'))
        for d in n.children_by_field_name('declarator'):
            key=declarator(d)
            if not key:continue
            if typ=='auto':
                val=field(d,'value') or field(n,'right')
                inferred=self.infer(val,env,f)
                if n.type=='for_range_loop' and inferred:
                    if 'map' in inferred:inferred='pair<'+self.cls_of(inferred)+'>'
                    elif self.cls_of(inferred):inferred=self.cls_of(inferred)+'::Ptr'
                if inferred:env[key]=inferred
            else:env[key]=f['owner']+'::Ptr' if typ=='Ptr' else typ

    def infer(self, n, env, f):
        if not n: return ''
        if n.type in ('identifier','field_identifier'): return env.get(txt(n), '')
        if n.type in ('parenthesized_expression','pointer_expression','cast_expression'):
            return self.infer(field(n,'argument') or field(n,'value') or (n.named_children[-1] if n.named_children else None),env,f)
        if n.type == 'field_expression':
            typ = self.infer(field(n,'argument'),env,f)
            if txt(field(n,'field'))=='second' and ('map' in typ or 'pair<' in typ):return self.cls_of(typ)+'::Ptr'
            return self.members(self.cls_of(typ)).get(txt(field(n,'field')), '')
        if n.type == 'subscript_expression':
            typ=self.infer(field(n,'argument'),env,f)
            return self.cls_of(typ)+'::Ptr' if self.cls_of(typ) else typ
        if n.type == 'call_expression':
            fun = field(n,'function'); name = txt(fun); args = arguments(field(n,'arguments'))
            if 'static_pointer_cast' in name and args: return self.infer(args[0],env,f)
            if name.endswith('::Create'): return name.split('::')[0]+'::Ptr'
            if fun.type == 'field_expression':
                short = txt(field(fun,'field')); typ = self.infer(field(fun,'argument'),env,f)
                if short in ('at','get','begin','front','back'):
                    return self.cls_of(typ)+'::Ptr' if self.cls_of(typ) else typ
                options = self.lookup(self.cls_of(typ)+'::'+short, len(args))
            else:
                options = self.lookup(f['owner']+'::'+name,len(args)) or self.lookup(name,len(args))
                if not options and f['name'].startswith('PIM_KERNEL::'):options=self.lookup('PIM_KERNEL::'+name,len(args))
            if options:
                result = self.functions[options[0]]['result']
                if result == 'Ptr': return self.functions[options[0]]['owner']+'::Ptr'
                return result
        return ''

    def condition_value(self, n):
        if not n: return None
        s = flat(n).replace('->','.')
        if s in self.constants: return self.constants[s]
        normalized=re.sub(r'^(?:device|scheduler)\.(model_config|system_config)\.',r'\1.',s)
        if normalized in self.constants:return self.constants[normalized]
        if s in self.member_aliases.get(self.active_owner,{}):return self.member_aliases[self.active_owner][s]
        if n.type in ('true','false','number_literal','string_literal'): return scalar(s)
        if n.type in ('parenthesized_expression','condition_clause'):
            return self.condition_value(field(n,'value') or (n.named_children[0] if n.named_children else None))
        if n.type == 'unary_expression':
            value = self.condition_value(field(n,'argument'))
            if value is not None and txt(field(n,'operator')) == '!': return not value
        if n.type == 'binary_expression':
            a,b = self.condition_value(field(n,'left')),self.condition_value(field(n,'right'))
            op = txt(field(n,'operator'))
            if op == '&&' and (a is False or b is False): return False
            if op == '||' and (a is True or b is True): return True
            if a is None or b is None: return None
            try:
                return {'==':lambda:a==b,'!=':lambda:a!=b,'<':lambda:a<b,'>':lambda:a>b,
                        '<=':lambda:a<=b,'>=':lambda:a>=b,'&&':lambda:bool(a and b),'||':lambda:bool(a or b)}[op]()
            except (KeyError, TypeError): return None
        # Main's YAML reads and simple string comparisons are configuration selectors.
        if n.type == 'call_expression':
            m = re.fullmatch(r'config((?:\["[^"]+"\])+)(?:\.as<[^>]+>\(\))',s)
            if m: return self.config.get('.'.join(re.findall(r'"([^"]+)"',m[1])))
            fun = field(n,'function')
            if fun and fun.type == 'field_expression' and txt(field(fun,'field')) == 'compare':
                receiver = txt(field(fun,'argument'))
                val = self.constants.get(receiver)
                if val is None:val=self.member_aliases.get(self.active_owner,{}).get(receiver)
                if receiver == 'processor_type': val = self.config.get('system.processor_type')
                args = arguments(field(n,'arguments'))
                if val is not None and args and args[0].type == 'string_literal': return 0 if val == scalar(txt(args[0])) else 1
        return None

    def resolve(self, n, f, env):
        """Return explicit targets; preserve uncertain dispatch as choices, never a guessed edge."""
        fun = field(n,'function') if n.type == 'call_expression' else field(n,'type')
        name = txt(fun); args_node = field(n,'arguments')
        args = arguments(args_node)
        argc = len(args)
        hint, dispatch = '', []
        if n.type == 'new_expression': return self.lookup(name+'::'+name,argc),hint,dispatch
        if fun and fun.type == 'parenthesized_expression':
            var = re.sub(r'[()*\s]', '', name)
            typ = self.cls_of(env.get(var,''))
            if not typ or not self.lookup(typ+'::operator()'): typ = 'Module'
            dispatch = self.dynamic_types(var, f, env, n.start_point.row+1)
            return self.lookup(typ+'::operator()',argc), '先进入 operator()，再按对象类型调用 forward。', dispatch
        if fun and fun.type == 'call_expression':
            table = next((k for k in self.dispatch_tables if name.startswith(k+'.at(')),None)
            if table:
                opts = []
                for processor, ids in self.dispatch_tables[table].items(): opts.extend(ids)
                return opts, '函数表分派；当前配置为 GPU，其他处理器实现保留为可选子流程。', []
        if fun and fun.type=='subscript_expression' and name.startswith('kernel['):
            opts=[]
            for short in ('Read_kernel','Write_kernel','GEMV'):opts.extend(self.lookup('PIM_KERNEL::'+short))
            return opts,'按 DRAMRequestType 选择已注册的命令生成内核：Read / Write / GEMV。',[]
        if name.startswith(('std::','YAML::','Ramulator::')): return [],'',[]
        if name in self.classes: return self.lookup(name+'::'+name,argc),hint,dispatch
        if fun and fun.type == 'field_expression':
            method = txt(field(fun,'field')); receiver = field(fun,'argument')
            typ = self.infer(receiver,env,f); cls = self.cls_of(typ)
            # A standard container's methods are not its element type's methods.
            if ('std::' in typ and any(k in typ for k in ('vector','map<','set<','list<','string'))) and receiver.type in ('identifier','field_expression'):
                return [],'',[]
            if method == 'forward' and cls == 'Module':
                opts = [i for i,v in self.functions.items() if v['short']=='forward' and v['owner'] in self.executable_types and v['owner'] in self.constructed_module_types]
                is_vec='vec' in txt(args[0]) if args else False
                opts=[i for i in opts if ('TensorVec' in self.functions[i]['param_types'])==is_vec]
                opts.sort(key=lambda i:(self.functions[i]['owner']!='Embedding',self.functions[i]['name']))
                return opts,'虚函数分派：由当前执行图记录所持的模块类型决定；各实现是候选分支。',[]
            if cls: return self.lookup(cls+'::'+method,argc),hint,dispatch
            # Known external Ramulator objects must not bind to same-named project functions.
            if txt(receiver) in ('frontend','memory_system'): return [],'',[]
            candidates = [i for i,v in self.functions.items() if v['short']==method and v['min_args']<=argc<=len(v['params'])]
            if candidates:
                self.unknown.append({'function':f['name'],'file':f['file'],'line':n.start_point.row+1,'call':flat(n),'candidates':[self.functions[i]['name'] for i in candidates]})
                return candidates,'接收对象类型无法静态唯一确定；下列定义为候选，请结合源码与调用上下文阅读。',[]
            return [],'',[]
        if name == 'forward' and f['owner']=='Module':
            opts = [i for i,v in self.functions.items() if v['short']=='forward' and v['owner']!='Module']
            # Match the Tensor/TensorVec overload when statically known.
            is_vec = 'TensorVec' in f['param_types']
            opts = [i for i in opts if ('TensorVec' in self.functions[i]['param_types']) == is_vec]
            return opts,'按父调用节点的实际模块类型分派；候选实现之间不是顺序调用。',[]
        opts = self.lookup(name,argc)
        if not opts and f['owner']: opts = self.lookup(f['owner']+'::'+name,argc)
        if not opts and f['name'].startswith('PIM_KERNEL::'): opts = self.lookup('PIM_KERNEL::'+name,argc)
        return opts,hint,dispatch

    def dynamic_types(self, var, f, env, line):
        cls = self.cls_of(env.get(var,''))
        if cls and cls != 'Module' and self.lookup(cls+'::forward'): return [cls]
        body = self.sources[f['file']].splitlines()[f['line']-1:line]
        preceding = '\n'.join(body)
        # Last lookup gives the dynamic module name for a named local variable.
        lookups = list(re.finditer(r'\b'+re.escape(var)+r'\s*=\s*get_module\("([^"]+)"',preceding))
        if lookups:
            name = lookups[-1].group(1)
            if name.startswith('MoE_decoder_'): return ['MoEDecoder']
            if name.startswith('decoder_'):
                # The first LLM loop selects either decoder class inside a conditional.
                if f['owner']=='LLM' and 'if ((model_config.model_name == "deepseekV3"))' not in preceding:return ['Decoder','MoEDecoder']
                return ['Decoder']
            types = self.module_types.get(f['owner']+'::'+name,[])
            return types
        if var=='top_module' and f['owner']=='Model': return ['LLM']
        return []

    def call_step(self,n,f,env,override=None):
        env=self.environment_at(f,n)
        targets,hint,dispatch = override or self.resolve(n,f,env)
        targets = [t for t in targets if (not self.functions[t]['trivial'] or self.functions[t]['short']=='forward') and self.functions[t]['short'] not in ('assertTrue','fail')]
        if not targets: return None
        # Resolve overloads with distinguishable Tensor / TensorVec arguments.
        args = field(n,'arguments')
        if len(targets)>1 and len({self.functions[t]['name'] for t in targets})==1 and arguments(args):
            for index,arg in enumerate(arguments(args)):
                actual=self.infer(arg,env,f)
                params=[self.functions[t]['param_types'][index] for t in targets if len(self.functions[t]['param_types'])>index]
                if actual and any('TensorVec' in p for p in params) and any('TensorVec' not in p for p in params):
                    actual_vec='TensorVec' in actual or ('vector' in actual and self.cls_of(actual)=='Tensor')
                    matches=[t for t in targets if len(self.functions[t]['param_types'])>index and actual_vec==('TensorVec' in self.functions[t]['param_types'][index])]
                    if matches:targets=matches
        line=n.start_point.row+1
        return dict(kind='call',id=f['id']+':'+str(n.start_byte),line=line,end=n.end_point.row+1,
            code=flat(n),targets=targets,hint=hint,dispatch=dispatch)

    def expression_calls(self,n,f,env):
        if not n or n.type=='comment': return []
        # Short-circuit and ternary operands retain their conditional execution.
        if n.type=='binary_expression' and txt(field(n,'operator')) in ('&&','||'):
            left,right=field(n,'left'),field(n,'right'); op=txt(field(n,'operator'))
            first=self.expression_calls(left,f,env); rest=self.expression_calls(right,f,env)
            if rest:
                v=self.condition_value(left)
                first.append(dict(kind='condition',id=f['id']+':sc'+str(n.start_byte),line=n.start_point.row+1,end=n.end_point.row+1,
                    code=flat(left),value=bool(v) if v is not None else None,then=rest if op=='&&' else [],otherwise=rest if op=='||' else [],hint='短路求值：只有满足前项条件才执行后项调用。'))
            return first
        if n.type=='conditional_expression':
            cond=field(n,'condition'); out=self.expression_calls(cond,f,env)
            out.append(dict(kind='condition',id=f['id']+':ternary'+str(n.start_byte),line=n.start_point.row+1,end=n.end_point.row+1,
                code=flat(cond),value=self.condition_value(cond),then=self.expression_calls(field(n,'consequence'),f,env),otherwise=self.expression_calls(field(n,'alternative'),f,env)))
            return out
        result=[]
        for c in n.named_children: result.extend(self.expression_calls(c,f,env))
        if n.type in ('call_expression','new_expression'):
            step=self.call_step(n,f,env)
            if step: result.append(step)
        if n.type=='delete_expression':
            local=self.environment_at(f,n)
            operand=n.named_children[-1] if n.named_children else None
            cls=self.cls_of(self.infer(operand,local,f))
            if cls:
                step=self.call_step(n,f,local,(self.lookup(cls+'::~'+cls,0),'delete 先调用项目定义的析构函数，再释放对象；递归删除的循环体只展示一次。',[]))
                if step:result.append(step)
        if n.type in ('assignment_expression','binary_expression'):
            op=txt(field(n,'operator'));local=self.environment_at(f,n)
            cls=self.cls_of(self.infer(field(n,'left'),local,f))
            if cls and self.lookup(cls+'::operator'+op,1):
                step=self.call_step(n,f,local,(self.lookup(cls+'::operator'+op,1),'项目定义的运算符重载。',[]))
                if step:result.append(step)
        # Direct object construction: Model model(...), SystemConfig system_config.
        if n.type=='declaration':
            typ=txt(field(n,'type'))
            if typ in self.classes:
                for d in n.children_by_field_name('declarator'):
                    val=field(d,'value')
                    if d.type=='function_declarator':
                        # tree-sitter cannot disambiguate a local object from a declaration
                        # using identifiers alone; verify every argument is a known variable.
                        params=arguments(field(d,'parameters'))
                        local=self.environment_at(f,n)
                        if params and all(flat(p) in local for p in params):
                            step=self.call_step(d,f,env,(self.lookup(typ+'::'+typ,len(params)),'直接构造对象。',[]))
                            if step:step['code']=typ+' '+flat(d);result.append(step)
                        continue
                    if val and val.type not in ('argument_list','initializer_list'): continue
                    if '*' in txt(d) or '&' in txt(d) or any(x.type=='function_declarator' for x in walk(d)): continue
                    argc=len(arguments(val))
                    step=self.call_step(d,f,env,(self.lookup(typ+'::'+typ,argc),'对象初始化，进入项目定义的构造函数。',[]))
                    if step: step['code']=typ+' '+flat(d); result.append(step)
        return result

    def exits_function(self,n):
        if not n:return False
        if n.type in ('return_statement','throw_statement'):return True
        if n.type in ('compound_statement','else_clause'):return any(self.exits_function(c) for c in n.named_children)
        if n.type=='if_statement':
            value=self.condition_value(field(n,'condition'))
            if value is not None:return self.exits_function(field(n,'consequence' if bool(value) else 'alternative'))
            return self.exits_function(field(n,'consequence')) and self.exits_function(field(n,'alternative'))
        return False

    def statements(self,n,f,env):
        if not n: return []
        typ=n.type; line=n.start_point.row+1
        base=dict(id=f['id']+':'+str(n.start_byte),line=line,end=n.end_point.row+1)
        if typ=='comment':return []
        if typ in ('compound_statement','else_clause','try_statement','catch_clause'):
            result=[];tail=[];exited=False
            for c in n.named_children:
                if c.type=='catch_clause':
                    steps=self.statements(field(c,'body'),f,env)
                    result.append(dict(base,kind='condition',id=f['id']+':catch'+str(c.start_byte),code='发生异常，进入 catch',value=None,then=steps,otherwise=[]))
                else:
                    steps=self.statements(c,f,env)
                    (tail if exited else result).extend(steps)
                    if self.exits_function(c):exited=True
            if tail:result.append(dict(base,kind='alternative',id=f['id']+':tail'+str(n.start_byte),code='当前配置已在上方返回；下面是其他路径的后续流程',body=tail))
            return result
        if typ=='if_statement':
            cond=field(n,'condition'); first=self.expression_calls(cond,f,env)
            yes=self.statements(field(n,'consequence'),f,env); no=self.statements(field(n,'alternative'),f,env)
            if not yes and not no:return []
            first.append(dict(base,kind='condition',code=flat(cond),value=self.condition_value(cond),then=yes,otherwise=no))
            return first
        if typ in ('for_statement','for_range_loop','while_statement','do_statement'):
            body=field(n,'body'); inner=self.statements(body,f,env)
            if not inner:return []
            cond=field(n,'condition'); init=field(n,'initializer') or field(n,'right')
            setup=self.expression_calls(init,f,env)
            checks=self.expression_calls(cond,f,env)
            update=self.expression_calls(field(n,'update'),f,env)
            header=txt(n).split('{',1)[0].strip()
            if body and body.start_byte>n.start_byte:header=n.text[:body.start_byte-n.start_byte].decode().strip()
            return setup+[dict(base,kind='loop',code=re.sub(r'\s+',' ',header),body=(inner+checks+update if typ=='do_statement' else checks+inner+update),hint='循环体展示一次；运行时可重复或零次。循环内保留判断与提前退出。')]
        if typ=='switch_statement':
            cond=field(n,'condition'); first=self.expression_calls(cond,f,env)
            cases=[]
            for c in field(n,'body').named_children:
                if c.type=='case_statement':
                    children=[]
                    for k in c.named_children:
                        if k!=field(c,'value'):children.extend(self.statements(k,f,env))
                    cases.append(dict(label='case '+txt(field(c,'value')) if field(c,'value') else 'default',body=children))
            return first+[dict(base,kind='switch',code=flat(cond),cases=cases)]
        result=self.expression_calls(n,f,env)
        if typ in ('return_statement','break_statement','continue_statement','throw_statement'):
            result.append(dict(base,kind='return' if typ=='return_statement' else 'control',code=flat(n)))
        return result

    def build_function(self,f):
        self.active_owner=f['owner']
        env=self.env_for(f); initial=[];initial_groups=[]
        init_list=next((n for n in f['node'].named_children if n.type=='field_initializer_list'),None)
        class_data=self.classes.get(f['owner'],{})
        order=class_data.get('bases',[])+list(class_data.get('members',{}))
        explicit={txt(n.named_children[0]) for n in init_list.named_children} if init_list else set()
        if init_list:
            members=self.members(f['owner'])
            # C++ initializes bases, then members in declaration order, not textual initializer order.
            inits=sorted(init_list.named_children,key=lambda n:order.index(txt(n.named_children[0])) if txt(n.named_children[0]) in order else len(order))
            for n in inits:
                if not n.named_children:continue
                name=txt(n.named_children[0]); cls=name if name in self.classes else self.cls_of(members.get(name,''))
                args=next((c for c in n.named_children if c.type in ('argument_list','initializer_list')),None)
                group=self.expression_calls(args,f,env)
                if cls and not ('::Ptr' in members.get(name,'') or '_ptr' in members.get(name,'') or 'std::' in members.get(name,'')):
                    vals=arguments(args)
                    if len(vals)==1 and self.cls_of(self.infer(vals[0],env,f))==cls:
                        initial_groups.append((order.index(name) if name in order else len(order),group))
                        continue  # implicit copy construction, not the unrelated normal constructor
                    step=self.call_step(n,f,env,(self.lookup(cls+'::'+cls,len(vals)),'构造函数体之前：先初始化基类，再按成员声明顺序初始化成员。',[]))
                    if step:group.append(step)
                initial_groups.append((order.index(name) if name in order else len(order),group))
        if f['short']==f['owner']:
            for name,typ in class_data.get('members',{}).items():
                if name in explicit or typ not in self.classes:continue
                targets=[t for t in self.lookup(typ+'::'+typ,0) if self.functions[t]['min_args']==0]
                if targets:
                    path,line=class_data['locations'][name]
                    step=dict(kind='call',id=f['id']+':implicit:'+name,line=f['line'],end=f['line'],code='隐式成员初始化：'+name+' → '+typ+'()',targets=targets,dispatch=[],
                        hint=f'C++ 隐式调用：进入构造函数体前，默认构造成员 {name}；成员声明位于 {path}:{line}。')
                    initial_groups.append((order.index(name),[step]))
        for _,group in sorted(initial_groups,key=lambda x:x[0]):initial.extend(group)
        steps=initial+self.statements(field(f['node'],'body'),f,env)
        if not steps:
            steps=[dict(kind='note',id=f['id']+':body',line=f['line'],end=f['end'],code=f['summary'] or '完成本函数的赋值、计算或标准库操作；没有需要展开的项目内部调用。')]
        return {k:f[k] for k in ('id','name','owner','short','file','line','end','summary','signature')} | {'steps':steps}

    def build(self):
        funcs={k:self.build_function(f) for k,f in self.functions.items()}
        hashes={p:hashlib.sha256(s.encode()).hexdigest() for p,s in self.sources.items()}
        hashes['config.yaml']=hashlib.sha256((ROOT/'config.yaml').read_bytes()).hexdigest()
        return dict(version=1,entry=self.lookup('main')[0],config=self.config,constants=self.constants,
            functions=funcs,sources=self.sources,hashes=hashes,root=str(ROOT),
            notes={
                'Model::Model':'构图阶段：创建每台设备上的模型模块，并通过 operator() / forward 传播形状、记录执行图；此时正式性能估算尚未开启。',
                'Cluster::run':'正式执行阶段：开启性能估算，重置执行图，然后轮转推进节点/设备。图上的循环折叠不改变设备之间的同步依赖。',
                'Scheduler::hittingQueue':'预热在源码中执行 10000 次，仅推进调度状态。图中展示循环体一次；之后仿真的初态仍是完整预热后的状态。',
                'Device::Device':'use_ramulator=off 时仍创建 DRAMInterface；是否逐请求调用 Ramulator 由后续执行路径决定。',
                'DRAMInterface::DRAMInterface':'这里先调用第三方 Ramulator 的配置解析和工厂，创建、连接 frontend 与 memory_system；这些外部实现不展开。之后才进入下方的项目自有内核注册。',
                'Device::execution':'perform_execution 随阶段改变：Model 构图期间为 false；Cluster::run 开启后为 true。只有正式执行阶段进入 Executor。',
                'ModuleGraph::run':'这里根据执行图记录调用具体模块的 forward。不同模块是循环中可能遇到的对象类型；候选实现不代表依次执行所有分支。',
                'Executor::executePType':'根据 LayerType 选择算子分支，再通过函数表选择 GPU / LOGIC / PIM 实现。',
                'Cluster::runIterationMixed':'CSV 写出在每轮开头；后面依次分配 token、执行、统计、更新请求并补充队列。'
            },unresolved=self.unknown)


def iterate_steps(steps):
    for step in steps:
        yield step
        for key in ('then','otherwise','body'):
            yield from iterate_steps(step.get(key,[]))
        for case in step.get('cases',[]):yield from iterate_steps(case['body'])


def validate(data, current=True):
    errors=[]; calls=0
    for fid,f in data['functions'].items():
        lines=data['sources'][f['file']].splitlines()
        if not (1<=f['line']<=f['end']<=len(lines)):errors.append('Invalid definition: '+f['name'])
        for s in iterate_steps(f['steps']):
            if not (1<=s['line']<=s['end']<=len(lines)):errors.append('Invalid location: '+s['id'])
            if s['kind']=='call':
                calls+=1
                if not s['targets']:errors.append('Empty call: '+s['id'])
                for t in s['targets']:
                    if t not in data['functions']:errors.append('Broken target: '+t)
    if current:
        for path,h in data['hashes'].items():
            if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=h:errors.append('Source changed: '+path)
    if errors:raise SystemExit('\n'.join(errors))
    print(f"Validated {len(data['functions'])} definitions, {calls} call sites, {len(data['sources'])} source files.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Validate saved graph against source hashes; no dependency needed')
    args=parser.parse_args()
    if args.check:
        validate(json.loads((HERE/'flow-data.json').read_text()));return
    data=Index().build()
    validate(data)
    (HERE/'flow-data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (HERE/'flow-data.js').write_text('window.FLOW_DATA = '+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';\n')
    print(f"Unresolved receiver call sites for review: {len(data['unresolved'])}")


if __name__=='__main__':main()
