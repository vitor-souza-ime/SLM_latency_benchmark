#!/usr/bin/env python3
"""Analyze measured benchmark data and plot scientific comparisons."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

METRICS={'e2e_s':'End-to-end latency (s)', 'ttfc_s':'Time to first visible content (s)',
         'throughput_after_ttfc_tps':'Throughput after first visible content (tokens/s)'}

def bootstrap_median(values, repeats=2000,seed=89):
    a=np.asarray(values,dtype=float)
    a=a[np.isfinite(a)]
    if not len(a):return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed)
    bs=np.median(rng.choice(a,size=(repeats,len(a)),replace=True),axis=1)
    return (float(np.median(a)),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5)))

def numeric(df):
    for c in ['e2e_s','ttfc_s','first_stream_event_s','post_ttfc_s','input_tokens',
              'output_tokens','reasoning_tokens','throughput_output_tps',
              'throughput_after_ttfc_tps','correct','ollama_eval_tps']:
        if c in df:df[c]=pd.to_numeric(df[c],errors='coerce')
    return df

def holm(p):
    vals=sorted(enumerate(p),key=lambda x:x[1]);out=[np.nan]*len(p)
    for rank,(i,value) in enumerate(vals):out[i]=min(1.,(len(p)-rank)*value)
    prev=0
    for i,_ in vals:
        out[i]=max(prev,out[i]);prev=out[i]
    return out

def make_box(data,metric,filename,title):
    labels=[];vals=[]
    for (name,env),grp in data.groupby(['model','environment'],sort=True):
        a=grp[metric].dropna().to_numpy()
        if len(a): labels.append(name+'\n'+env);vals.append(a)
    if not vals:return
    fig,ax=plt.subplots(figsize=(max(10,1.65*len(labels)),6))
    ax.boxplot(vals,showfliers=True,showmeans=True)
    ax.set_xticks(range(1,len(labels)+1),labels,rotation=20,ha='right')
    ax.set_ylabel(METRICS[metric]);ax.set_title(title)
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(filename,dpi=180);plt.close(fig)

def accuracy_chart(df,path):
    qa=df[df.task_type=='accuracy'].copy()
    if qa.empty:return
    acc=qa.pivot_table(index='category',columns=['model','environment'],values='correct',aggfunc='mean')
    if acc.empty:return
    fig,ax=plt.subplots(figsize=(max(10,acc.shape[1]*1.55),5.5))
    image=ax.imshow(100*acc.to_numpy(dtype=float),aspect='auto',vmin=0,vmax=100,cmap='Greys')
    ax.set_xticks(np.arange(acc.shape[1]),[f'{a}\n{b}' for a,b in acc.columns],rotation=28,ha='right')
    ax.set_yticks(np.arange(acc.shape[0]),acc.index)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            v=acc.iat[i,j]
            if pd.notna(v):ax.text(j,i,f'{100*v:.0f}%',ha='center',va='center',color=('white' if v<.45 else 'black'))
    ax.set_title('Multiple-choice accuracy by category (%)')
    fig.colorbar(image,ax=ax,label='Accuracy (%)',shrink=.8)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def scatter_chart(df,path):
    qa=df[df.task_type=='accuracy'].copy()
    ag=qa.groupby(['model','environment'],as_index=False).agg(accuracy=('correct','mean'),latency=('e2e_s','median'))
    if ag.empty:return
    fig,ax=plt.subplots(figsize=(9,6))
    for _,r in ag.iterrows():
        ax.scatter(r.latency,r.accuracy*100,s=100)
        ax.annotate(f'{r.model} ({r.environment})',(r.latency,r.accuracy*100),xytext=(5,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('Median QA latency (s)');ax.set_ylabel('QA accuracy (%)')
    ax.set_title('Observed latency–accuracy relationship')
    ax.set_ylim(-3,103);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def cdf_chart(perprompt,path):
    fig,ax=plt.subplots(figsize=(10,6))
    count=0
    for (model,env),grp in perprompt.groupby(['model','environment'],sort=True):
        v=np.sort(grp.e2e_s.dropna().to_numpy(dtype=float))
        if not len(v):continue
        ax.step(v,np.arange(1,len(v)+1)/len(v),where='post',label=model+' · '+env)
        count+=1
    if not count:plt.close(fig);return
    ax.set_xlabel('Median per-prompt latency (s)')
    ax.set_ylabel('Fração acumulada de perguntas')
    ax.set_title('Empirical CDF of per-prompt latency')
    ax.legend(fontsize=8);ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def paired_results(perprompt):
    rows=[]
    for (model,kind),group in perprompt.groupby(['model','task_type']):
        wide=group.pivot_table(index='id',columns='environment',values='e2e_s')
        if not {'local','remote'}.issubset(wide.columns):continue
        wide=wide[['local','remote']].dropna()
        if len(wide)==0:continue
        ratios=wide['local']/wide['remote']
        ratio,lo,hi=bootstrap_median(ratios)
        p=np.nan
        if len(wide)>=5 and (np.log(ratios)!=0).any():
            try:p=float(wilcoxon(np.log(ratios)).pvalue)
            except ValueError:pass
        rows.append(dict(model=model,task_type=kind,n_paired_prompts=len(wide),
                         remote_speedup_median=ratio,speedup_ci_lo=lo,speedup_ci_hi=hi,
                         wilcoxon_p=p,lat_local_median=float(wide['local'].median()),
                         lat_remote_median=float(wide['remote'].median())))
    return pd.DataFrame(rows)

def speedup_chart(pairs,path):
    if pairs.empty:return
    pairs=pairs.reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(10,max(4,len(pairs)*.65+1.3)))
    y=np.arange(len(pairs))
    x=pairs.remote_speedup_median.to_numpy()
    low=(x-pairs.speedup_ci_lo.to_numpy()).clip(0)
    high=(pairs.speedup_ci_hi.to_numpy()-x).clip(0)
    ax.errorbar(x,y,xerr=[low,high],fmt='o',capsize=4)
    ax.axvline(1,linestyle='--',linewidth=1,label='Equal latency')
    ax.set_yticks(y,[f'{r.model} · {r.task_type}' for _,r in pairs.iterrows()])
    ax.invert_yaxis();ax.set_xlabel('Remote speedup = local latency / remote latency')
    ax.set_title('Paired prompt comparisons: medians and 95% bootstrap CIs')
    ax.grid(axis='x',alpha=.25)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if not args.input.exists():parser.error('Input CSV does not exist')
    dest=args.output or args.input.parent/'figures'
    dest.mkdir(parents=True,exist_ok=True)
    df=numeric(pd.read_csv(args.input,encoding='utf-8-sig'))
    all_count=len(df)
    good=df[df.status=='ok'].copy()
    good=good[np.isfinite(good.e2e_s)&(good.e2e_s>0)]
    if good.empty:
        print('No valid observations. Run the benchmark before analyzing.');return
    summary=[]
    for (model,env,kind),g in good.groupby(['model','environment','task_type']):
        rows={'model':model,'environment':env,'task_type':kind,'n_success':len(g),
              'n_questions':g.id.nunique(),'latency_p95_s':float(np.percentile(g.e2e_s,95)),
              'mean_output_tokens':float(g.output_tokens.mean()),
              'reasoning_present_pct':float(100*g.output_reasoning.notna().mean())}
        # Bootstrap prompt medians, so repetitions are not treated as independent prompts.
        gp=g.groupby('id',as_index=False).agg({'e2e_s':'median','ttfc_s':'median',
                    'throughput_after_ttfc_tps':'median','throughput_output_tps':'median'})
        for metric in ['e2e_s','ttfc_s','throughput_after_ttfc_tps']:
            md,lo,hi=bootstrap_median(gp[metric])
            rows[metric+'_median']=md;rows[metric+'_ci_lo']=lo;rows[metric+'_ci_hi']=hi
        if kind=='accuracy':
            rows['accuracy_pct']=float(100*g.correct.mean())
            byitem=g.groupby('id').correct.mean()
            _,lo,hi=bootstrap_median(byitem) # Bootstrap the mean for accuracy rather than its median.
            rng=np.random.default_rng(42)
            vals=byitem.to_numpy();bs=rng.choice(vals,size=(2000,len(vals)),replace=True).mean(axis=1)
            rows['accuracy_ci_lo']=float(np.percentile(bs,2.5)*100)
            rows['accuracy_ci_hi']=float(np.percentile(bs,97.5)*100)
        summary.append(rows)
    summ=pd.DataFrame(summary)
    summ.to_csv(dest/'summary.csv',index=False,encoding='utf-8-sig')
    gp=(good.groupby(['model','environment','task_type','id'],as_index=False)
          .agg({'e2e_s':'median','ttfc_s':'median','throughput_after_ttfc_tps':'median'}))
    gp.to_csv(dest/'per_prompt_medians.csv',index=False,encoding='utf-8-sig')
    make_box(gp,'e2e_s',dest/'01_latency_boxplots.png','End-to-end latency per prompt (medians of repetitions)')
    make_box(gp,'ttfc_s',dest/'02_ttfc_boxplots.png','Time to first visible content per prompt')
    perf=gp[gp.task_type=='performance']
    if not perf.empty:
        make_box(perf,'throughput_after_ttfc_tps',dest/'03_throughput_boxplots.png',
                 'Observed throughput for generation prompts (client estimate)')
    accuracy_chart(good,dest/'04_accuracy_heatmap.png')
    scatter_chart(good,dest/'05_quality_latency_scatter.png')
    cdf_chart(gp,dest/'06_latency_cdf.png')
    pairs=paired_results(gp)
    if not pairs.empty:
        # Holm correction within each task-type family.
        pairs['wilcoxon_p_holm']=np.nan
        for kind,g in pairs.groupby('task_type'):
            idx=g.index[g.wilcoxon_p.notna()].tolist()
            if idx:pairs.loc[idx,'wilcoxon_p_holm']=holm(pairs.loc[idx,'wilcoxon_p'].tolist())
        pairs.to_csv(dest/'paired_speedup.csv',index=False,encoding='utf-8-sig')
        speedup_chart(pairs,dest/'07_paired_speedup.png')
    errors=df[df.status!='ok']
    if not errors.empty:
        errors.groupby(['model','environment','http_status'],dropna=False).size().rename('count').reset_index().to_csv(dest/'errors.csv',index=False)
    report=[]
    report.append('# Automated report — GPU-local vs remote SLM benchmark\n')
    report.append(f'Observations: {all_count}; valid: {len(good)}; failed: {len(errors)}.\n')
    report.append('## Per-model, environment, and task-type metrics\n')
    report.append(summ.round(3).to_string(index=False))
    if not pairs.empty:
        report.append('\n## Paired per-prompt comparison (local vs remote latency)\n')
        report.append(pairs.round(4).to_string(index=False))
    report.append('\n## Interpretation and limitations\n')
    report.append('- TTFC measures first visible text, not necessarily the first internally generated token; hidden reasoning can precede visible output.')
    report.append('- E2E latency includes networking, gateway and provider queuing, tokenization, prefill, decoding, and transfers; it is not accelerator-only runtime.')
    report.append('- Client-estimated tokens/s after TTFC can be biased by streamed chunk boundaries and hidden reasoning tokens.')
    report.append('- Local quantization and remote backend configuration can differ; pin and document remote providers.')
    report.append('- QA accuracy uses strict A–D letter matching; verbose or malformed answers count as incorrect.')
    report.append('- Bootstrap confidence intervals resample prompts, not within-prompt repetitions as independent examples.')
    report.append('- Report truncations, caching, provider changes, and HTTP 429 errors; rerun affected sessions as needed.')
    report.append('- Do not generalize these findings to all SLMs or all cloud providers.')
    (dest/'report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('Figures and statistics:',dest)
    print('Valid records:',len(good),' | Failed:',len(errors))
    print(summ[['model','environment','task_type','n_success','e2e_s_median','ttfc_s_median']].to_string(index=False))

if __name__=='__main__':
    main()
