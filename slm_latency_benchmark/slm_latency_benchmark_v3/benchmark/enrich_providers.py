#!/usr/bin/env python3
"""Fetch actual OpenRouter provider and cost metadata AFTER inference, to avoid measurement interference."""
import argparse, os, time
from pathlib import Path
import pandas as pd
import requests


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--pause',type=float,default=0.3)
    args=p.parse_args()
    key=os.environ.get('OPENROUTER_API_KEY','')
    if not key:p.error('Set OPENROUTER_API_KEY.')
    df=pd.read_csv(args.input,encoding='utf-8-sig',dtype={'generation_id':'string'})
    dest=args.input.parent/'provider_metadata.csv'
    ids=df.loc[(df.environment=='remote') & df.generation_id.notna(),'generation_id'].dropna().unique()
    if not len(ids):
        print('No OpenRouter generation IDs found in CSV.');return
    sess=requests.Session()
    rows=[]
    for i,genid in enumerate(ids,1):
        row={'generation_id':genid,'provider_name':'','quantization':'','total_cost':'',
             'native_tokens_prompt':'','native_tokens_completion':'','error':''}
        try:
            r=sess.get('https://openrouter.ai/api/v1/generation',params={'id':genid},
                       headers={'Authorization':'Bearer '+key},timeout=25)
            r.raise_for_status()
            d=r.json().get('data') or {}
            for k in ['provider_name','total_cost','native_tokens_prompt','native_tokens_completion']:
                row[k]=d.get(k,'')
            row['quantization']=d.get('quantization','')
        except (requests.RequestException,ValueError) as exc:
            row['error']=str(exc)[:300]
        rows.append(row)
        if i%30==0:print(i,'/',len(ids),'looked up')
        time.sleep(max(0,args.pause))
    pd.DataFrame(rows).to_csv(dest,index=False,encoding='utf-8-sig')
    print('Saved:',dest)

if __name__=='__main__':main()
