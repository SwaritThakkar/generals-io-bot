#!/usr/bin/env python3
"""Node-count and route-quality comparison for the A* heuristic variants."""
from __future__ import annotations
import json, pathlib, sys, time
import jax, numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import genmod
from genmod import EnvConfig
from genmod.bots import expander_bot
from server.bot_protocol import BotState
from strategies.wastar import search as base_search
from strategies.wastar_timegrowth import search as tg_search

def to_bot(st,cfg):
    owner,army=np.asarray(st.owner),np.asarray(st.army)
    land=np.asarray([(owner==p).sum() for p in range(2)],np.int32)
    totals=np.asarray([army[owner==p].sum() for p in range(2)],np.int32)
    return BotState(cfg.height,cfg.width,2,int(st.step),owner,army,
        np.asarray(st.mountain),np.asarray(st.castle),np.asarray(st.general),
        np.asarray(st.general_idx),np.asarray(st.alive),land,totals,
        {'maxTicks':cfg.max_steps,'generalGrowthInterval':2,'landGrowthInterval':50})

def sample_states(cfg,seeds):
    out=[]
    for seed in seeds:
        st=genmod.reset(jax.random.PRNGKey(seed),cfg)
        wait_side=seed%2
        for tick in range(101):
            if tick in (50,75,100): out.append((seed,tick,wait_side,to_bot(st,cfg)))
            actions=[0,0]
            actions[1-wait_side]=int(expander_bot(jax.random.PRNGKey(seed*101+tick),st,cfg,player=1-wait_side))
            st,_,done,_=genmod.step(st,np.asarray(actions,np.int32),cfg)
            if bool(done):break
    return out

def main():
    cfg=EnvConfig(height=23,width=23,max_steps=300,bfs_iters=92)
    stats=[];seeds=range(93000,93012)
    for seed,tick,wait_side,st in sample_states(cfg,seeds):
        for side in (0,1):
            specs=[('base_a1',base_search,1.0,{}),('tg_a1',tg_search,1.0,{}),
                   ('tg_a15',tg_search,1.5,{}),
                   ('tg_payback',tg_search,1.0,{'goal_mode':'payback','max_labels_per_cell':6})]
            for name,fn,eps,kw in specs:
                t=time.perf_counter();p=fn(st,side,epsilon=eps,**kw);dt=time.perf_counter()-t
                stats.append({'seed':seed,'tick':tick,'wait_side':wait_side,'side':side,
                    'variant':name,'seconds':dt,'target':list(p.target) if p and p.target else None,
                    'target_kind':p.target_kind if p else None,'ticks':p.ticks if p else None,
                    'expanded':p.expanded if p else 0,'generated':p.generated if p else 0,
                    'army_at_goal':getattr(p,'army_at_target',None) if p else None})
    out=ROOT/'models'/'tournaments'/'astar_heuristic_node_stats.jsonl';out.parent.mkdir(exist_ok=True)
    out.write_text(''.join(json.dumps(x,separators=(',',':'))+'\n' for x in stats))
    for name in ('base_a1','tg_a1','tg_a15','tg_payback'):
        xs=[x for x in stats if x['variant']==name]
        print(name,'n=',len(xs),'expanded_avg=',round(np.mean([x['expanded'] for x in xs]),1),
              'generated_avg=',round(np.mean([x['generated'] for x in xs]),1),
              'seconds_avg_ms=',round(1000*np.mean([x['seconds'] for x in xs]),3),
              'feasible_target_pct=',round(100*np.mean([x['target'] is not None for x in xs]),1))
    print('saved',out)
if __name__=='__main__':main()
