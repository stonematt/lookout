import sys, numpy as np, pandas as pd, itertools
from prep import load, hourly, O
pd.set_option('display.width',250); pd.set_option('display.max_rows',500)
def wy(d): return d.year+1 if d.month>=10 else d.year
def build(mac, strict=False):
    r,_=load(mac); h=hourly(r)
    h=h.copy(); h['day']=[i.date() for i in h.index]
    ok=h.n>=(6 if strict else 1)
    h['cnt']=ok; h['w']=h.wet&ok
    d=h.groupby('day').agg(wet_h=('w','sum'),cnt_h=('cnt','sum'),rain=('rain','sum'))
    d.index=pd.to_datetime(d.index)
    full=pd.date_range(d.index.min(),d.index.max(),freq='D')
    d=d.reindex(full); d[['wet_h','cnt_h','rain']]=d[['wet_h','cnt_h','rain']].fillna(0)
    d['known']=d.cnt_h>=18   # day countable: >=18 countable hours
    d['wetday']=(d.rain>=0.0095)&d.known
    return r,h,d
def sh(a,b): return a/b.replace(0,np.nan)
def run(name,mac):
    out=open(O+f'{name}_report.txt','w')
    def P(*a,**k):
        print(*a,file=out)
    r,h,d=build(mac); _,_,d6=build(mac,True)
    P(name,'days',len(d),d.index.min().date(),d.index.max().date(),'hours',len(h),'known days',d.known.sum())
    P('hours with >=1 reading',int(h.cnt.sum()),' >=6 readings',int(d6.cnt_h.sum()),' wet hours',int(h.w.sum()),'wet hours strict',int(d6.wet_h.sum()))
    # tips
    rises=r[(r.flag>0)]
    P('\n== 3 drizzle: wet-hour tip size distribution')
    wh=h[h.w]
    P('wet hours with exactly 1 tip (0.01):',(wh.flag.round(3)==0.01).mean().round(3),' <=0.02:',(wh.flag<=0.0205).mean().round(3),' >0.10:',(wh.flag>0.1).mean().round(3))
    gaps=np.diff(rises.t.astype('int64').to_numpy())/6e10
    P('minutes between consecutive tip-readings (all pairs): quantiles',np.percentile(gaps,[10,25,50,75,90]).round(1))
    inper=gaps[gaps<=180]
    P('pairs <=180min (same rainy period) n=',len(inper),'quantiles p10/25/50/75/90',np.percentile(inper,[10,25,50,75,90]).round(1))
    for lim in [5,10,15,30,60,120,180]: P(f'  share of in-period gaps <= {lim}min: {(inper<=lim).mean():.3f}')
    P('  tip-readings:',len(rises),' readings with rise >0.01 (multi-tip in 5 min):',(rises.flag>0.0105).mean().round(3))
    # weekly
    wk=d.copy(); wk['wk']=wk.index.to_period('W-SUN').start_time
    W=wk.groupby('wk').agg(wet_h=('wet_h','sum'),cnt_h=('cnt_h','sum'),rain=('rain','sum'),days=('rain','size'))
    W['wet_share']=(W.wet_h/W.cnt_h.replace(0,np.nan)).round(3); W['rain']=W.rain.round(2)
    W.to_csv(O+f'{name}_weekly_all.csv'); 
    end=d.index.max(); W365=W[W.index>end-pd.Timedelta(days=365)]
    W365.to_csv(O+f'{name}_weekly_last365.csv')
    P('\n== 1 weekly last 365 (see csv) weeks',len(W365),'; wet share quantiles p10/50/90 of weekly',W365.wet_share.quantile([.1,.5,.9]).round(3).tolist())
    P(W365.to_string())
    # monthly by WY
    m=d.copy(); m['wy']=[wy(x) for x in m.index]; m['mo']=m.index.month
    M=m.groupby(['wy','mo']).agg(w=('wet_h','sum'),c=('cnt_h','sum'),rain=('rain','sum'),nd=('rain','size'))
    M['share']=M.w/M.c.replace(0,np.nan)
    pv=(M.share.unstack('mo').reindex(columns=[10,11,12,1,2,3,4,5,6,7,8,9])*100).round(1)
    pv.to_csv(O+f'{name}_monthly_share_by_wy.csv')
    pr=M.rain.unstack('mo').reindex(columns=[10,11,12,1,2,3,4,5,6,7,8,9]).round(2); pr.to_csv(O+f'{name}_monthly_rain_by_wy.csv')
    P('\n== 2 monthly wet share % by WY (cols Oct..Sep)'); P(pv.to_string()); P('rain in'); P(pr.to_string())
    wyr=m.groupby('wy').agg(w=('wet_h','sum'),c=('cnt_h','sum'),rain=('rain','sum'),wetdays=('wetday','sum'),known=('known','sum'))
    wyr['share']=(wyr.w/wyr.c).round(4); P(wyr.round(2).to_string())
    # item 6
    d['s1']=d.wet_h/d.cnt_h.replace(0,np.nan)
    c7=d.cnt_h.rolling(7).sum(); w7=d.wet_h.rolling(7).sum(); d['s7']=w7/c7.replace(0,np.nan)
    d['mo']=d.index.month
    P('\n== 6 meter values (share %, p10/p50/p90, mean, %days with zero)')
    for lab,ms in [('Dec-Feb',[12,1,2]),('Jun-Aug',[6,7,8])]:
        for k in ['s1','s7']:
            x=d[d.mo.isin(ms)&d.known][k].dropna()*100
            P(lab,k,len(x),x.quantile([.1,.5,.9]).round(1).tolist(),'mean',round(x.mean(),1),'zero',round((x==0).mean(),2))
    # wet-hour counts per day too
    for lab,ms in [('Dec-Feb',[12,1,2]),('Jun-Aug',[6,7,8])]:
        x=d[d.mo.isin(ms)&d.known].wet_h; P(lab,'wet hours/day p10/50/90',x.quantile([.1,.5,.9]).tolist(),'wetday frac',round(d[d.mo.isin(ms)&d.known].wetday.mean(),3))
    # dry spell lengths
    daily=d.rain.where(d.known)
    from lookout.core.rainfall_analysis import dry_spells
    ds=dry_spells(daily.set_axis(daily.index))
    ds['mo']=pd.to_datetime(ds.first_day).dt.month
    P('\ncompleted dry spells n',len(ds),'median',ds.days.median())
    P('dry spells >=7 days starting Nov-Mar:'); P(ds[(ds.days>=7)&ds.mo.isin([11,12,1,2,3])].to_string())
    ds.to_csv(O+f'{name}_dry_spells.csv')
    d.to_csv(O+f'{name}_daily.csv')
    # rules
    sim(name,d,ds,P)
    out.close()
def known(d,i): return d.known.iloc[i]
def simulate(d,entry,exit_,warm=14):
    """entry(i)->bool, exit_(i)->bool evaluated with data through day i-1; state for day i"""
    n=len(d); st=[False]*n; cur=False
    for i in range(n):
        if i>=warm:
            if not cur and entry(i): cur=True
            elif cur and exit_(i): cur=False
        st[i]=cur
    return pd.Series(st,index=d.index)
def sim(name,d,ds,P):
    n=len(d); idx=d.index
    wet=d.wet_h.to_numpy(float); cnt=d.cnt_h.to_numpy(float); wd=d.wetday.to_numpy(); kn=d.known.to_numpy()
    cs_w=np.concatenate([[0],np.cumsum(wet)]); cs_c=np.concatenate([[0],np.cumsum(cnt)])
    def share(i,k):  # trailing k days ending i-1
        a=max(0,i-k); c=cs_c[i]-cs_c[a]
        return (cs_w[i]-cs_w[a])/c if c>0 else np.nan
    def nwet(i,k): return int(wd[max(0,i-k):i].sum())
    def dryrun(i):
        k=0
        while i-1-k>=0 and kn[i-1-k] and not wd[i-1-k]: k+=1
        return k
    month=idx.month.to_numpy()
    rules={}
    rules['a_calendar']=pd.Series(((month>=10)|(month<=5)),index=idx)
    for ti in [.05,.10,.15]:
        for to in [.01,.02,.05]:
            rules[f'b_in{int(ti*100)}_out{int(to*100)}']=simulate(d,lambda i,ti=ti:(share(i,7)>=ti),lambda i,to=to:(share(i,14)<to) if not np.isnan(share(i,14)) else False)
    for N in [2,3,4]:
        for D in [7,10,14]:
            rules[f'c_N{N}_D{D}']=simulate(d,lambda i,N=N:nwet(i,7)>=N,lambda i,D=D:dryrun(i)>=D)
            def ent(i,N=N): return nwet(i,7)>=(N+1 if month[i] in (6,7,8,9) else N)
            def ex(i,D=D): return dryrun(i)>=(D+7 if (month[i]>=10 or month[i]<=5) else D)
            rules[f'd_N{N}_D{D}']=simulate(d,ent,ex)
    last=idx.max()-pd.Timedelta(days=365)
    rows=[]; det=[]
    for k,s in rules.items():
        f=s[s!=s.shift(1)].iloc[1:]  # flips (first value is initial)
        flips=[(dt.date(),'dry->wet' if v else 'wet->dry') for dt,v in f.items() if (dt-idx.min()).days>=14]
        wyc={}
        for dt,_ in flips: wyc[wy(dt)]=wyc.get(wy(dt),0)+1
        ds_=[x[0] for x in flips]
        flap=[(a,b,'->'.join([fa[1]])) for (a,fa),(b,fb) in zip(zip(ds_,flips),zip(ds_[1:],flips[1:])) if (b-a).days<7]
        summer=[(x,t) for x,t in flips if t=='dry->wet' and x.month in (6,7,8,9)]
        sept=[(x,t) for x,t in flips if t=='dry->wet' and x.month in (9,)]
        winterdry=[(x,t) for x,t in flips if t=='wet->dry' and x.month in (12,1,2)]
        l365=[(str(x),t) for x,t in flips if pd.Timestamp(x)>last]
        rows.append(dict(rule=k,flips=len(flips),flips_by_wy=str(dict(sorted(wyc.items()))),flaps_lt7d=len(flap),summer_Jun_Sep_entries=len(summer),sept_entries=len(sept),winter_DecFeb_exits=len(winterdry),flips_last365=len(l365),wet_day_share=round(s.mean(),3)))
        det.append((k,flips,flap,l365,summer,winterdry))
    R=pd.DataFrame(rows); R.to_csv(O+f'{name}_rule_summary.csv',index=False)
    P('\n== 4 rule backtest summary'); P(R.to_string())
    with open(O+f'{name}_rule_flips.txt','w') as fh:
        for k,flips,flap,l365,summer,winterdry in det:
            fh.write(f'## {k}\nflips: {[(str(a),b) for a,b in flips]}\nflaps<7d: {[(str(a),str(b)) for a,b,_ in flap]}\nJun-Sep entries: {[(str(a)) for a,_ in summer]}\nDec-Feb exits: {[str(a) for a,_ in winterdry]}\nlast365: {l365}\n\n')
    # the calendar rule agreement: when do rules disagree with calendar
    cal=rules['a_calendar']
    P('\nflips last365 for each rule:')
    for k,flips,flap,l365,summer,winterdry in det: P(k,l365)
    P('\nagreement with calendar (share of days same state):')
    for k,s in rules.items(): P(k,round((s==cal).mean(),3),end=' | ')
if __name__=='__main__':
    run('lookout','98-CD-AC-22-0D-E5'); run('salmon','24-4C-AB-74-5D-4D')
