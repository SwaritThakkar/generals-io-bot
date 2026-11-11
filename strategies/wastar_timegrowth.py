"""Time-dependent resource-constrained weighted A* for Generals.

Compared with ``strategies.wastar``, this planner treats moving army as a
resource and charges every hostile/neutral garrison encountered along the
route. It projects deterministic production on hostile tiles at arrival. Its
heuristic remains Manhattan time plus a target-value penalty; epsilon controls
the usual weighted-A* speed/quality tradeoff.
"""
from __future__ import annotations

import heapq
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from genmod.config import DIRS, NUM_CELL_ACTIONS  # noqa: E402

V_ENEMY_GENERAL=200.0
V_ENEMY_CASTLE=40.0
V_NEUTRAL_CASTLE=25.0
V_ENEMY_TILE=3.0
V_NEUTRAL_TILE=1.0
VALUE_TO_TICKS=1.0


@dataclass
class Plan:
    epsilon: float
    action: int
    mover: tuple[int,int]
    dest: tuple[int,int]
    direction: int
    target: tuple[int,int] | None
    target_kind: str
    path: list[tuple[int,int]]=field(default_factory=list)
    expanded: int=0
    generated: int=0
    cost: float=0.0
    ticks: int=0
    army_at_target: int=0
    army_spent: int=0


def _boards(state):
    return (np.asarray(state.owner,np.int8),np.asarray(state.army,np.int32),
            np.asarray(state.mountain,bool),np.asarray(state.castle,bool),
            np.asarray(state.general,bool))


def _classify(r,c,owner,castle,general,me,opp):
    if general[r,c] and owner[r,c]==opp: return "enemy general",V_ENEMY_GENERAL
    if castle[r,c] and owner[r,c]==opp: return "enemy city",V_ENEMY_CASTLE
    if castle[r,c] and owner[r,c]<0: return "neutral city",V_NEUTRAL_CASTLE
    if owner[r,c]==opp: return "enemy tile",V_ENEMY_TILE
    return "neutral tile",V_NEUTRAL_TILE


def _income(owner,castle,general,cell,side,now,end):
    r,c=cell
    if owner[r,c]!=side or end<=now: return 0
    fast=bool(castle[r,c] or general[r,c])
    return ((end//2-now//2) if fast else 0)+(end//50-now//50)


def search(state,player:int,epsilon:float=1.0,opp:int|None=None,
           single_goal:bool=True,max_labels_per_cell:int=0,
           goal_mode:str="linear"):
    """Find a feasible route; unlike the old planner, pay garrisons en route.

    Labels at each cell are Pareto states ``(elapsed_ticks, remaining_army)``.
    An earlier/stronger label dominates a later/weaker one. Enemy/castle/land
    production is projected only until combat arrival because combat resolves
    before the new tick's production.
    """
    owner,army,mountain,castle,general=_boards(state)
    h,w=owner.shape; me=int(player); opp=1-me if opp is None else int(opp)
    now=int(getattr(state,"tick",getattr(state,"step",0)))
    starts=list(zip(*np.nonzero((owner==me)&(army>=2))))
    if not starts: return None
    # Match the macro objective set of the existing A*: castles or enemy
    # territory. Neutral land remains traversable and its garrison is paid,
    # but it should not mask a feasible producer/general objective.
    goals=(owner!=me)&~mountain&(castle|(owner==opp))
    if not goals.any(): return None
    values=np.zeros((h,w),np.float64); kinds={}
    for r,c in zip(*np.nonzero(goals)):
        kind,val=_classify(int(r),int(c),owner,castle,general,me,opp)
        kinds[(int(r),int(c))]=kind; values[r,c]=val
    vmax=float(values[goals].max())
    rr,cc=np.indices((h,w))
    if goal_mode not in ("linear","payback"):
        raise ValueError("goal_mode must be 'linear' or 'payback'")
    penalty=(np.where(goals,VALUE_TO_TICKS*(vmax-values),np.inf)
             if goal_mode=="linear" else np.where(goals,0.0,np.inf))
    hmap=np.full((h,w),np.inf,np.float64)
    for r,c in zip(*np.nonzero(goals)):
        np.minimum(hmap,np.abs(rr-r)+np.abs(cc-c)+penalty[r,c],out=hmap)

    # Label arrays are append-only so parent links remain stable. Per-cell
    # frontiers retain labels with a different time/army trade-off.
    labels=[]; fronts=[[[] for _ in range(w)] for _ in range(h)]
    heap=[]; tie=0; generated=expanded=0
    for sr0,sc0 in starts:
        sr,sc=int(sr0),int(sc0)
        idx=len(labels); labels.append([sr,sc,0,int(army[sr,sc]),-1,(sr,sc),True])
        fronts[sr][sc].append(idx)
        tie+=1; heapq.heappush(heap,(epsilon*hmap[sr,sc],0,-int(army[sr,sc]),tie,idx))
        generated+=1
    found=None; best_goal_score=-np.inf
    while heap:
        _f,tick,_negfuel,_tie,idx=heapq.heappop(heap)
        r,c,elapsed,fuel,parent,origin,active=labels[idx]
        if not active: continue
        expanded+=1
        if goals[r,c]:
            if goal_mode=="linear":
                found=idx; break
            remaining=max(0,int(getattr(state,"config",{}).get("maxTicks",400))-now-elapsed)
            kind,_value=_classify(r,c,owner,castle,general,me,opp)
            if kind=="enemy general":
                goal_score=1e9+fuel
            elif kind=="neutral city":
                goal_score=25.0+0.50*remaining+0.02*fuel-0.30*elapsed
            elif kind=="enemy city":
                goal_score=40.0+0.50*remaining+0.02*fuel-0.30*elapsed
            else:
                goal_score=3.0+0.02*remaining+0.02*fuel-0.30*elapsed
            if goal_score>best_goal_score:
                best_goal_score=goal_score;found=idx
            if kind=="enemy general": break
        for d,(dr,dc) in enumerate(DIRS):
            nr,nc=r+dr,c+dc
            if not (0<=nr<h and 0<=nc<w) or mountain[nr,nc]: continue
            nt=elapsed+1
            # Moving leaves one army behind at the previous tile.
            nf=fuel-1
            side=int(owner[nr,nc])
            if side!=me:
                # The target stays under its current owner up to arrival;
                # combat precedes production on the arrival tick.
                arrival_tick=now+nt-1
                defense=int(army[nr,nc])
                if side>=0:
                    defense+=_income(owner,castle,general,(nr,nc),side,now,arrival_tick)
                nf-=defense
                if nf<=0: continue
            if nf<1: continue
            old=fronts[nr][nc]
            dominated=False
            for oi in old:
                ol=labels[oi]
                if ol[6] and ol[2]<=nt and ol[3]>=nf:
                    dominated=True; break
            if dominated: continue
            for oi in old:
                ol=labels[oi]
                if ol[6] and nt<=ol[2] and nf>=ol[3]: ol[6]=False
            ni=len(labels); labels.append([nr,nc,nt,nf,idx,origin,True])
            fronts[nr][nc].append(ni)
            if max_labels_per_cell>0:
                live=[li for li in fronts[nr][nc] if labels[li][6]]
                if len(live)>max_labels_per_cell:
                    # Keep the best resource/time trade-offs, pruning only
                    # dominated/weak search labels (not a policy move).
                    live.sort(key=lambda li:(labels[li][3]-0.15*labels[li][2]),reverse=True)
                    for li in live[max_labels_per_cell:]: labels[li][6]=False
            tie+=1
            heapq.heappush(heap,(nt+epsilon*hmap[nr,nc],nt,-nf,tie,ni))
            generated+=1
    if found is None:return None
    path=[]; idx=found
    while idx>=0:
        r,c=labels[idx][0],labels[idx][1];path.append((r,c));idx=labels[idx][4]
    path.reverse()
    if len(path)<2:return None
    (r0,c0),(r1,c1)=path[:2];d=DIRS.index((r1-r0,c1-c0))
    action=(r0*w+c0)*NUM_CELL_ACTIONS+1+d
    goal=path[-1]; elapsed=labels[found][2]; fuel=labels[found][3]
    return Plan(float(epsilon),int(action),(r0,c0),(r1,c1),d,goal,kinds[goal],path,
                expanded,generated,elapsed+penalty[goal],elapsed,fuel,
                int(army[r0,c0])-fuel)


act=lambda state,player: (lambda p: int(p.action) if p else 0)(search(state,player,epsilon=1.25))
