import sys, json, pandas as pd, numpy as np
sys.path.insert(0,'/Users/mstone/src/github.com/stonematt/lookout')
from lookout.core.rain_corrections import correct_counters
D='/Users/mstone/src/github.com/stonematt/lookout/data/'
O='/Users/mstone/.claude/jobs/f618a35c/tmp/backtest/'
TZ='America/Los_Angeles'
def load(mac):
    df=pd.read_parquet(D+mac+'.parquet')
    cor=(json.load(open(D+mac+'.rain_corrections.json')) or {}).get('corrections') or []
    df=df.sort_values('dateutc',ascending=False).reset_index(drop=True)
    df=correct_counters(df,cor)
    df=df.sort_values('dateutc').drop_duplicates('dateutc').reset_index(drop=True)
    t=pd.to_datetime(df.dateutc,unit='ms',utc=True).dt.tz_convert(TZ)
    v=df.dailyrainin.astype(float).to_numpy()
    d=np.diff(v,prepend=np.nan)
    rain=np.where(d>=0,d,v); rain[0]=0
    gap=np.diff(df.dateutc.to_numpy(),prepend=0)/60000
    rain=np.where(np.isnan(rain),0,rain)
    r=pd.DataFrame({'t':t,'v':v,'rain':rain,'gap':gap})
    r['flag']=np.where(r.gap<=30,r.rain,0.0)
    r['reset']=(d<0)
    return r,len(cor)
def hourly(r):
    h=r.t.dt.tz_convert('UTC').dt.floor('h').dt.tz_convert(TZ)
    g=r.assign(h=h).groupby('h').agg(n=('rain','size'),flag=('flag','sum'),rain=('rain','sum'))
    g['wet']=g.flag>=0.0095
    return g
def daily(h):
    x=h.copy(); x['day']=x.index.tz_localize(None).normalize() if x.index.tz is None else pd.DatetimeIndex([i.date() for i in x.index])
    return x
if __name__=='__main__':
    for mac,nm in [('98-CD-AC-22-0D-E5','lookout'),('24-4C-AB-74-5D-4D','salmon')]:
        r,nc=load(mac); h=hourly(r)
        print(nm,len(r),nc,'corrections; resets',r.reset.sum(),'gap>30 readings',(r.gap>30).sum(),'rain in gap-rises',r.rain[(r.gap>30)].sum(),'total',r.rain.sum())
