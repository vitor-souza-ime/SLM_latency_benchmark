#!/usr/bin/env python3
"""Optional English Streamlit UI for the GPU-local vs remote SLM latency benchmark."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from benchmark.catalog import matched_providers

ROOT = Path(__file__).resolve().parent
MODELS = json.loads((ROOT / 'models.json').read_text(encoding='utf-8'))
QA = json.loads((ROOT / 'data/questions.json').read_text(encoding='utf-8'))
GEN = json.loads((ROOT / 'data/generation_prompts.json').read_text(encoding='utf-8'))
COMPACT = json.loads((ROOT / 'data/compact_protocol.json').read_text(encoding='utf-8'))

st.set_page_config(page_title='SLM Latency Benchmark', page_icon='⏱️', layout='wide')
st.title('SLM Latency Benchmark')
st.caption('GPU-local Ollama (RTX 4070) vs OpenRouter · English prompts · real measurements only')
st.info('Primary metric: client-observed end-to-end response latency. No CPU inference experiments.')

with st.sidebar:
    st.header('Experiment settings')
    selections = st.multiselect('Models',
        options=list(range(1, len(MODELS)+1)),default=list(range(1, len(MODELS)+1)),
        format_func=lambda n: MODELS[n-1]['name'])
    mode = st.radio('Environments', ['Both (paired)', 'GPU local only', 'OpenRouter only'])
    profile = st.radio('Dataset', ['compact', 'full'], horizontal=True)
    repeats = st.number_input('Repetitions per prompt', min_value=1,max_value=50,value=2,step=1)
    warmups = st.number_input('Warmups per model and environment', min_value=0,max_value=10,value=1)
    cap = st.number_input('Maximum allowed remote requests',min_value=0,max_value=9999,value=100)
    st.checkbox('Omit unavailable models in BOTH environments', value=True, key='skip_unavailable')
    key = st.text_input('OpenRouter API key (not saved)', type='password', value=os.getenv('OPENROUTER_API_KEY',''))
    accepted = st.checkbox('I understand remote API calls may be billed.')

ids=','.join(map(str,selections))
tasks = [t for t in QA+GEN if (profile=='full' or t['id'] in
          set(COMPACT['question_ids']+COMPACT['generation_ids']))]
remote = (mode != 'GPU local only')
local = (mode != 'OpenRouter only')
measured=len(selections)*len(tasks)*int(repeats)
warmed=len(selections)*int(warmups)

s1,s2,s3,s4=st.columns(4)
s1.metric('English prompts',len(tasks))
s2.metric('Selected models',len(selections))
s3.metric('GPU-local measured requests',measured if local else 0)
s4.metric('Remote requests incl. warmups',(measured+warmed) if remote else 0)

with st.expander('Model mapping and dataset preview',expanded=False):
    st.dataframe(pd.DataFrame([{'Model':m['name'],'Ollama':m['ollama'],
        'OpenRouter':m['openrouter']} for m in MODELS if (MODELS.index(m)+1) in selections]),
        hide_index=True,use_container_width=True)
    st.dataframe(pd.DataFrame([{'ID':t['id'],'Type':t['type'],'Group':t['category'],
        'Prompt':t['prompt'],'Expected':t.get('expected') or '—'} for t in tasks]),
        hide_index=True,use_container_width=True)

left,right=st.columns([1,1])
with left:
    if st.button('Check remote model endpoints (no inference charge)', use_container_width=True,
                 disabled=not selections):
        for i in selections:
            m=MODELS[i-1]
            try:
                providers=matched_providers(m['openrouter'],m.get('provider',''))
                if providers:
                    st.success(f'{m["name"]}: {len(providers)} eligible endpoint(s)')
                else:
                    st.error(f'{m["name"]}: no active generation endpoints. Exclude it or enable omission.')
            except Exception as exc:
                st.error(f'{m["name"]}: availability check error: {exc}')

command=[sys.executable,'-u',str(ROOT/'benchmark'/'run.py'),
         '--model-ids',ids,'--profile',profile,'--repeats',str(repeats),
         '--warmups',str(warmups),'--max-remote-requests',str(cap)]
if mode=='GPU local only':command.append('--local-only')
if mode=='OpenRouter only':command.append('--remote-only')
if st.session_state.get('skip_unavailable'):command.append('--skip-unavailable')

with right:
    st.code('python benchmark/run.py '+ ' '.join(command[3:]),language='bash')
    if (measured+warmed if remote else 0) > cap:
        st.error('Remote request budget exceeded. Reduce repetitions or change the cap intentionally.')
    if remote and not key:
        st.warning('Enter an OpenRouter API key before starting paid/remote work.')
    if remote and not accepted:
        st.warning('Confirm potential billing before starting.')

st.divider()
st.subheader('Run benchmark')
can_run=bool(selections) and ((not remote) or ((measured+warmed)<=cap and bool(key) and accepted))
if st.button('Run selected experiment',type='primary',disabled=not can_run):
    env=os.environ.copy()
    if remote:env['OPENROUTER_API_KEY']=key
    lines=[]
    log=st.empty()
    st.warning('Keep this application open until the experiment finishes. Measurements are saved to CSV as they complete.')
    proc=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,bufsize=1,text=True)
    try:
        for line in proc.stdout:
            lines.append(line.rstrip('\n'))
            log.code('\n'.join(lines[-28:]),language='text')
        rc=proc.wait()
        if rc==0:st.success('Experiment completed. Select the new result below and generate figures.')
        else:st.error(f'Experiment stopped (exit code {rc}). No successful inference was fabricated.')
    except Exception as exc:
        proc.terminate()
        st.error(f'Execution stopped: {exc}')

st.divider()
st.subheader('Recorded results and figures')
result_files=sorted((ROOT/'results').glob('*/raw_results.csv'),reverse=True)
if not result_files:
    st.write('No experimental data yet. Run a benchmark above or use the command-line tools.')
else:
    choices={str(p.parent.name):p for p in result_files}
    name=st.selectbox('Experiment folder',list(choices))
    result_path=choices[name]
    st.caption(str(result_path))
    data=pd.read_csv(result_path,encoding='utf-8-sig')
    st.dataframe(data[['model','environment','id','repetition','status','e2e_s','ttfc_s',
                       'output_tokens','correct']].head(150),use_container_width=True)
    if st.button('Generate English figures from this result'):
        completed=subprocess.run([sys.executable,str(ROOT/'benchmark'/'analyze.py'),
                                  '--input',str(result_path)],capture_output=True,text=True)
        if completed.returncode:
            st.error(completed.stderr or completed.stdout)
        else:st.success(completed.stdout)
    fig_dir=result_path.parent/'figures'
    if fig_dir.exists():
        for image_file in sorted(fig_dir.glob('*.png')):
            st.image(str(image_file),caption=image_file.stem,use_container_width=True)
