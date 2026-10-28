#!/usr/bin/env python3
"""GPU-only local Ollama vs remote OpenRouter benchmark. Real client-side timings; no fabricated measurements."""
import argparse, csv, datetime as dt, json, os, platform, random, re, subprocess, sys, time
from pathlib import Path
import requests
try:
    from benchmark.catalog import matched_providers
except ModuleNotFoundError:
    from catalog import matched_providers

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['timestamp_utc','model','model_id','environment','provider_requested','generation_id',
          'id','task_type','category','language','repetition','expected','answer_text','prompt',
          'output_text','output_reasoning','correct','status','error','finish_reason',
          'e2e_s','ttfc_s','first_stream_event_s','last_content_s','post_ttfc_s',
          'input_tokens','output_tokens','reasoning_tokens','token_count_source',
          'throughput_output_tps','throughput_after_ttfc_tps',
          'ollama_total_s','ollama_load_s','ollama_prompt_eval_s','ollama_eval_s',
          'ollama_eval_tps','http_status','gpu_memory_used_mib']


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def mem_gpu():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=5,check=True)
        return ';'.join(x.strip() for x in r.stdout.splitlines() if x.strip())
    except (OSError,subprocess.SubprocessError):
        return ''


def machine_metadata(url):
    out=dict(utc=now_iso(),python=sys.version,platform=platform.platform(),processor=platform.processor(),
             cpu_count=os.cpu_count(),ollama_host=url)
    try:
        import psutil
        out['memory_ram_gb']=round(psutil.virtual_memory().total/1e9,2)
        out['cpu_brand']=platform.uname().processor
    except ImportError:
        pass
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                         capture_output=True,text=True,check=True,timeout=5)
        out['gpu_info']=p.stdout.strip()
    except (OSError,subprocess.SubprocessError):
        out['gpu_info']='not detected (may be CPU / AMD / Apple Silicon)'
    try:
        r=requests.get(url+'/api/version',timeout=3)
        out['ollama_version']=r.json().get('version') if r.ok else None
        tags=requests.get(url+'/api/tags',timeout=8)
        out['ollama_models']=[{'name':m.get('name'),'digest':m.get('digest'),
                              'size':m.get('size'),'details':m.get('details')}
                             for m in tags.json().get('models',[])] if tags.ok else []
    except requests.RequestException as e:
        out['ollama_connection_error']=str(e)
    return out


def normalize_mc(text):
    """Strict MCQ scoring: exactly one A-D letter, optionally followed by punctuation."""
    m = re.fullmatch(r'\s*([A-Da-d])\s*[.\)]?\s*', text or '')
    return m.group(1).upper() if m else ''


def stream_ollama(session, url, model, messages, max_tokens, seed, think, timeout):
    body={'model':model,'messages':messages,'stream':True,'keep_alive':'30m',
          'options':{'temperature':0,'seed':seed,'num_predict':max_tokens}}
    if think is not None:
        body['think']=think
    t0=time.perf_counter()
    with session.post(url+'/api/chat',json=body,stream=True,timeout=(15,timeout)) as response:
        response.raise_for_status()
        status=response.status_code
        output=[]; reasoning=[]; first_event=None; first_text=None; last_text=None; final={}
        for line in response.iter_lines():
            if not line:
                continue
            t=time.perf_counter()
            if first_event is None:
                first_event=t
            frame=json.loads(line)
            if frame.get('error'):
                raise RuntimeError(frame['error'])
            msg=frame.get('message') or {}
            content=msg.get('content') or ''
            if content:
                output.append(content)
                if first_text is None:
                    first_text=t
                last_text=t
            if msg.get('thinking'):
                reasoning.append(msg['thinking'])
            if frame.get('done'):
                final=frame
        end=time.perf_counter()
    ns=lambda k: round(final[k]/1e9,7) if isinstance(final.get(k),(int,float)) else None
    out_tokens=final.get('eval_count')
    return dict(output_text=''.join(output),output_reasoning=''.join(reasoning),
       e2e_s=end-t0, ttfc_s=(first_text-t0) if first_text else None,
       first_stream_event_s=(first_event-t0) if first_event else None,
       last_content_s=(last_text-t0) if last_text else None,
       post_ttfc_s=(end-first_text) if first_text else None,
       input_tokens=final.get('prompt_eval_count'),output_tokens=out_tokens,reasoning_tokens=None,
       token_count_source='ollama eval_count',
       ollama_total_s=ns('total_duration'),ollama_load_s=ns('load_duration'),
       ollama_prompt_eval_s=ns('prompt_eval_duration'),ollama_eval_s=ns('eval_duration'),
       ollama_eval_tps=((out_tokens/(final['eval_duration']/1e9)) if out_tokens and final.get('eval_duration') else None),
       finish_reason=final.get('done_reason'),http_status=status,generation_id='')


def stream_openrouter(session, key, model, messages, max_tokens, seed, think, provider, timeout):
    body={'model':model,'messages':messages,'stream':True,'temperature':0,
          'seed':seed,'max_tokens':max_tokens}
    if provider:
        body['provider']={'order':[provider],'allow_fallbacks':False,'require_parameters':True}
    # Controle solicitado, não garante que o provedor execute identicamente. Validar metadados.
    if think is not None:
        body['reasoning']={'enabled':bool(think)}
    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    t0=time.perf_counter()
    with session.post('https://openrouter.ai/api/v1/chat/completions',headers=headers,
                      json=body,stream=True,timeout=(20,timeout)) as response:
        if not response.ok:
            raise requests.HTTPError(f'HTTP {response.status_code}: {response.text[:500]}',response=response)
        status=response.status_code
        genid=response.headers.get('X-Generation-Id','')
        output=[]; reasoning=[]; first_event=None; first_text=None; last_text=None; usage={}; finish=None
        for line in response.iter_lines(decode_unicode=False):
            if not line:
                continue
            if isinstance(line,bytes):
                line=line.decode('utf-8','replace')
            line=line.strip()
            if line.startswith(':') or not line.startswith('data:'):
                continue
            payload=line[5:].strip()
            if payload=='[DONE]':
                break
            t=time.perf_counter()
            frame=json.loads(payload)
            if frame.get('error'):
                raise RuntimeError('OpenRouter stream error: '+str(frame['error'])[:400])
            if first_event is None:
                first_event=t
            genid=genid or frame.get('id','')
            if frame.get('usage'):
                usage=frame['usage']
            for choice in frame.get('choices') or []:
                delta=choice.get('delta') or {}
                c=delta.get('content') or ''
                if isinstance(c,str) and c:
                    output.append(c)
                    if first_text is None:
                        first_text=t
                    last_text=t
                rt=delta.get('reasoning') or ''
                if isinstance(rt,str) and rt:
                    reasoning.append(rt)
                if choice.get('finish_reason'):
                    finish=choice['finish_reason']
        end=time.perf_counter()
    details=usage.get('completion_tokens_details') or {}
    return dict(output_text=''.join(output),output_reasoning=''.join(reasoning),
       e2e_s=end-t0,ttfc_s=(first_text-t0) if first_text else None,
       first_stream_event_s=(first_event-t0) if first_event else None,
       last_content_s=(last_text-t0) if last_text else None,
       post_ttfc_s=(end-first_text) if first_text else None,
       input_tokens=usage.get('prompt_tokens'), output_tokens=usage.get('completion_tokens'),
       reasoning_tokens=details.get('reasoning_tokens'),token_count_source='openrouter usage',
       finish_reason=finish,http_status=status,generation_id=genid)


def load_tasks(limit_qa=0,limit_gen=0,profile='compact'):
    items=json.loads((ROOT/'data/questions.json').read_text(encoding='utf-8'))
    perf=json.loads((ROOT/'data/generation_prompts.json').read_text(encoding='utf-8'))
    if profile == 'compact':
        config=json.loads((ROOT/'data/compact_protocol.json').read_text(encoding='utf-8'))
        selected=set(config['question_ids']+config['generation_ids'])
        all_tasks=items+perf
        tasks=[task for task in all_tasks if task['id'] in selected]
        if len(tasks)!=len(selected) or len(selected)!=12:
            raise ValueError('compact_protocol.json is invalid: 12 unique IDs are required.')
        return tasks
    if profile != 'full':
        raise ValueError('Profile must be compact or full')
    if limit_qa: items=items[:limit_qa]
    if limit_gen: perf=perf[:limit_gen]
    return items+perf


def verify_full_gpu(model_id):
    """Require a loaded model's Ollama PROCESSOR field to show 100% GPU."""
    try:
        proc = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, check=True, timeout=15)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'GPU verification failed (ollama ps): {exc}') from exc
    lines = proc.stdout.splitlines()
    matching = [line for line in lines if line.strip().lower().startswith(model_id.lower() + ' ')]
    if not matching or not any('100% gpu' in line.lower() for line in matching):
        raise RuntimeError('GPU-only requirement not satisfied. Run `ollama ps` and verify '
                           f'100% GPU for {model_id}. Current output: {proc.stdout[:1200]}')


def run():
    p = argparse.ArgumentParser(description='English-only, GPU-local vs OpenRouter inference latency benchmark.')
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--profile', choices=['compact','full'], default='compact',
                   help='compact: 12 English prompts; full: 72 English prompts.')
    p.add_argument('--limit-qa', type=int, default=0, help='Full profile only; 0 means all 60 QA prompts.')
    p.add_argument('--limit-gen', type=int, default=0, help='Full profile only; 0 means all 12 generation prompts.')
    p.add_argument('--local-only', action='store_true')
    p.add_argument('--remote-only', action='store_true')
    p.add_argument('--model-ids', default='1,2,3,4', help='Model IDs (1-4), comma-separated.')
    p.add_argument('--qa-max-tokens', type=int, default=8)
    p.add_argument('--gen-max-tokens', type=int, default=320)
    p.add_argument('--ollama-host', default=os.environ.get('OLLAMA_HOST','http://127.0.0.1:11434').rstrip('/'))
    p.add_argument('--pause-remote', type=float, default=2.0,
                   help='Seconds between remote requests; helpful for free-tier rate limits.')
    p.add_argument('--max-remote-requests', type=int, default=100,
                   help='Hard safety ceiling including remote warmups (default: 100).')
    p.add_argument('--dry-run', action='store_true', help='No API calls and no results created.')
    p.add_argument('--skip-unavailable', action='store_true',
                   help='Drop remote-unavailable models from BOTH environments to preserve valid pairs.')
    p.add_argument('--skip-remote-preflight', action='store_true',
                   help='Dangerous: bypass no-charge catalog availability verification.')
    p.add_argument('--allow-unverified-gpu', action='store_true',
                   help='Bypass Ollama GPU-only verification; never use for a scientific GPU-only campaign.')
    p.add_argument('--timeout', type=float, default=240)
    p.add_argument('--seed', type=int, default=20261026)
    p.add_argument('--output', default='', help='Results directory; defaults to results/<timestamp>.')
    args = p.parse_args()
    if args.local_only and args.remote_only:
        p.error('Choose only one of --local-only and --remote-only.')
    if args.repeats < 1 or args.warmups < 0 or args.max_remote_requests < 0:
        p.error('repeats must be >=1; warmups and max-remote-requests must be >=0.')
    if args.profile == 'compact' and (args.limit_qa or args.limit_gen):
        p.error('--limit-qa and --limit-gen require --profile full.')
    if min(args.qa_max_tokens,args.gen_max_tokens) < 1:
        p.error('Token caps must be positive.')
    all_models = json.loads((ROOT/'models.json').read_text(encoding='utf-8'))
    try:
        indices=[int(s.strip()) for s in args.model_ids.split(',')]
        if not indices or len(indices)!=len(set(indices)) or min(indices)<1 or max(indices)>len(all_models):
            raise ValueError('Invalid or duplicate model ID.')
        models=[all_models[i-1] for i in indices]
    except ValueError:
        p.error(f'--model-ids must contain unique integers in 1..{len(all_models)}.')
    tasks=load_tasks(args.limit_qa,args.limit_gen,args.profile)
    environments=['local'] if args.local_only else (['remote'] if args.remote_only else ['local','remote'])

    # The dry run is offline, deliberately excluding the no-charge online catalog preflight.
    if not args.dry_run and 'remote' in environments and not args.skip_remote_preflight:
        ready, unavailable = [], []
        for m in models:
            try:
                eps=matched_providers(m['openrouter'],m.get('provider') or '')
                if eps: ready.append(m)
                else: unavailable.append((m['name'],'no eligible generation endpoints'))
            except Exception as exc:
                unavailable.append((m['name'],str(exc)[:180]))
        if unavailable:
            for name, why in unavailable:
                print(f'UNAVAILABLE: {name}: {why}',flush=True)
            if args.skip_unavailable:
                models=ready
                print('Omitting unavailable models in BOTH environments to keep only valid pairs.',flush=True)
            else:
                p.error('Remote endpoint preflight failed. Zero inference requests sent. '
                        'Run benchmark/check_catalog.py or use --skip-unavailable.')
        if not models:
            p.error('No remote models available; zero inference requests sent.')

    measured_per_environment=len(models)*len(tasks)*args.repeats
    warmups_per_environment=len(models)*args.warmups
    remote_measured=measured_per_environment if 'remote' in environments else 0
    remote_warmups=warmups_per_environment if 'remote' in environments else 0
    remote_total=remote_measured+remote_warmups
    print(f'Profile: {args.profile} | prompts: {len(tasks)} | models: {len(models)} | repetitions: {args.repeats}',flush=True)
    print(f'Local measured: {measured_per_environment if "local" in environments else 0} | '
          f'Remote measured: {remote_measured} | Remote warmups: {remote_warmups} | '
          f'MAX POSSIBLE REMOTE REQUESTS: {remote_total}',flush=True)
    if remote_total>args.max_remote_requests:
        p.error(f'SAFETY BLOCK: {remote_total} remote requests exceed your cap of '
                f'{args.max_remote_requests}. Explicitly raise it with --max-remote-requests {remote_total}.')
    if args.dry_run:
        print('DRY RUN ONLY: no endpoint checks, no inference calls, and no experimental files.',flush=True)
        return
    key=os.environ.get('OPENROUTER_API_KEY','')
    if 'remote' in environments and not key:
        p.error('Set OPENROUTER_API_KEY in your environment; never store it in source code.')
    if 'local' in environments and not args.allow_unverified_gpu:
        if not mem_gpu():
            p.error('NVIDIA GPU not detected by nvidia-smi; GPU-only mode refuses CPU fallback.')
    dest=Path(args.output).expanduser().resolve() if args.output else (ROOT/'results'/dt.datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    if dest.exists() and (dest/'raw_results.csv').exists():
        p.error('The result CSV already exists. Choose another --output to avoid mixing campaigns.')
    dest.mkdir(parents=True,exist_ok=True)
    csv_path=dest/'raw_results.csv'
    meta=machine_metadata(args.ollama_host)
    meta['benchmark_arguments']=vars(args)
    meta['model_definitions']=models
    meta['task_ids']=[t['id'] for t in tasks]
    meta['remote_requests_budget']=remote_total
    meta['provider_unpinned_warning']=[m['name'] for m in models if not m.get('provider')]
    (dest/'experiment_config.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    if 'remote' in environments and meta['provider_unpinned_warning']:
        print('NOTE: Remote provider not pinned. For final paper, pin providers and document them.',flush=True)
    session=requests.Session()
    rng=random.Random(args.seed)
    # Rotate the order of models; within each model randomize local vs remote environment.
    blocks=[(m,env) for m in models for env in environments]
    rng.shuffle(blocks)
    num_planned=len(tasks)*args.repeats*len(blocks)
    print('Results directory:',dest,'| Planned measured inferences:',num_planned,flush=True)
    counter=0
    with csv_path.open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=FIELDS)
        writer.writeheader();file.flush()
        for model,env in blocks:
            slug=model['ollama'] if env=='local' else model['openrouter']
            provider=model.get('provider') or ''
            print(f'\n[{env}] {model["name"]} / {slug} | warmups={args.warmups}',flush=True)
            sysmsg=model.get('system_prompt') or 'Answer in English.'
            for w in range(args.warmups):
                task=tasks[w%len(tasks)]
                messages=[{'role':'system','content':sysmsg},{'role':'user','content':task['prompt']}]
                fn=stream_ollama if env=='local' else stream_openrouter
                try:
                    if env=='local':
                        fn(session,args.ollama_host,slug,messages,24,args.seed,model.get('think'),args.timeout)
                    else:
                        fn(session,key,slug,messages,24,args.seed,model.get('think'),provider,args.timeout)
                except Exception as exc:
                    raise RuntimeError(f'Warmup failure for {model["name"]} ({env}); stopping to avoid '
                                       f'additional failed requests: {exc}') from exc
                if env=='local' and not args.allow_unverified_gpu:
                    verify_full_gpu(slug)
                print(f'  Warmup {w+1}: OK',flush=True)
                if env=='remote':time.sleep(max(0,args.pause_remote))
            for rep in range(1,args.repeats+1):
                queue=tasks[:];rng.shuffle(queue)
                for task in queue:
                    counter+=1
                    row={k:'' for k in FIELDS}
                    row.update(timestamp_utc=now_iso(),model=model['name'],model_id=slug,
                               environment=env,provider_requested=provider,id=task['id'],
                               task_type=task['type'],category=task['category'],
                               language=task['language'],repetition=rep,
                               expected=task.get('expected') or '',
                               answer_text=task.get('answer_text') or '',prompt=task['prompt'])
                    messages=[{'role':'system','content':sysmsg},{'role':'user','content':task['prompt']}]
                    tokens=args.qa_max_tokens if task['type']=='accuracy' else args.gen_max_tokens
                    try:
                        if env=='local':
                            result=stream_ollama(session,args.ollama_host,slug,messages,tokens,
                                                 args.seed,model.get('think'),args.timeout)
                        else:
                            result=stream_openrouter(session,key,slug,messages,tokens,
                                                     args.seed,model.get('think'),provider,args.timeout)
                        row.update(result);row['status']='ok'
                        if task['type']=='accuracy':
                            row['correct']=int(normalize_mc(result['output_text'])==task['expected'])
                        n=result.get('output_tokens');tt=result.get('e2e_s');tail=result.get('post_ttfc_s')
                        row['throughput_output_tps']=(n/tt) if n and tt else ''
                        row['throughput_after_ttfc_tps']=((n-1)/tail) if n and n>1 and tail and tail>0 else ''
                    except Exception as exc:
                        row['status']='error';row['error']=str(exc)[:800]
                        row['http_status']=getattr(getattr(exc,'response',None),'status_code','')
                    if env=='local':row['gpu_memory_used_mib']=mem_gpu()
                    writer.writerow(row);file.flush()
                    if env=='remote' and row['status']=='error':
                        raise RuntimeError('Remote inference failed; stopping immediately to protect '
                                           f'the request budget. Partial results saved at {csv_path}. '
                                           f'Error: {row["error"]}')
                    if counter%10==0 or row['status']!='ok':
                        print(f'  {counter}/{num_planned} {task["id"]} {row["status"]} '
                              f'e2e={row["e2e_s"] if row["e2e_s"]!="" else "-"}',flush=True)
                    if env=='remote':time.sleep(max(0,args.pause_remote))
    print('\nComplete:',csv_path,flush=True)
    print('Create plots: python benchmark/analyze.py --input',csv_path,flush=True)

if __name__=='__main__':
    run()
