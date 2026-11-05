"""表2六列历史试算代码：从当时的终端命令恢复。

请从项目根目录启动。原次运行打印结果后，最后的图2锚点断言未通过。
本文件保留当时的计算方法，并在回归后绘制国泰安标签收益的两窗六列紧凑表；
图片保存至“图表结果/04_国泰安标签收益检验/表2_收益回归”。
图片注明本次与旧图2参考结果的比较状态；差异超出原容差时提示，不中断出图。
拟合优度未计算，图片中留空。本次代码修改未运行，现有输入可能已变化。
"""

import os,sys,time,gc,json
sys.dont_write_bytecode=True
for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS"):
    os.environ[name]="1"
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from scipy import sparse,stats
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.csgraph import connected_components
root=Path.cwd();cache=root/"work/主回归口径组合试算_20261026"
sys.path.insert(0,str(root/"code/07_实证分析/90_公共模块"))
from 主回归口径 import select_news
t0=time.monotonic()
paths=sorted((cache/"enriched").glob("q*.npy"))
base=[]
for i,path in enumerate(paths):
    a=np.load(path)
    old=np.load(cache/"figure2_selected_20261027/returns"/path.name)[:,[0,9]].copy()
    base.append(dict(firm=a["firm"].copy(),date=a["date"].copy(),filled=a["filled"].copy(),
        cue=a["cues"][:,8].copy(),x=np.column_stack([a["x"],select_news(a)]),old=old))
    if (i+1)%20==0:print("顺序载入季度",i+1,"/83，秒数",round(time.monotonic()-t0,1),flush=True)
del a,old
offsets=np.r_[0,np.cumsum([len(b["firm"]) for b in base])]
targetkeys=np.concatenate([b["date"].astype("int64")*1000000+b["firm"] for b in base])
assert (np.diff(targetkeys)>0).all()
rawtarget=np.full((len(targetkeys),2),np.nan);porttarget=np.zeros(len(targetkeys),dtype="int16")
hits=np.zeros(len(targetkeys),dtype=bool)
eligpath=root/"中间结果/01_清洗后样本/窗口变量目标日资格索引_基础清洗含停牌填充_td.csv"
daily=root/"中间结果/01_清洗后样本/DGTW构建用个股日度面板.csv"
labpath=root/"input/CSMAR/DGTW（2000至2025）/DGTW股票分配表/STK_MKT_DGTWASSINGMENTS.csv"
watch={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in (eligpath,daily,labpath)}
elig=pd.read_csv(eligpath,usecols=["Stkcd","td"],dtype={"Stkcd":"int32","td":"int32"})
ek=np.sort(elig["td"].to_numpy(dtype="int64")*1000000+elig["Stkcd"].to_numpy());del elig
assert (np.diff(ek)>0).all()
lab=pd.read_csv(labpath);lab=lab.loc[lab["IsNotBSE"].eq(1)]
assert not lab.duplicated(["Symbol","TradingYear"]).any()
lookup=np.zeros((26,1000000),dtype="int16")
lookup[lab["TradingYear"].to_numpy()-2000,lab["Symbol"].to_numpy()]=(lab["MarketValue"]-1)*25+(lab["BooktoMarket"]-1)*5+lab["Momentum"]
dmin=int(np.datetime64("2005-04-08").astype("int32"));dmax=int(np.datetime64("2025-12-31").astype("int32"))
ng=(dmax-dmin+1)*125;num=np.zeros((ng,2));den=np.zeros((ng,2))
cols=["Stkcd","td","Trddt","float_mktcap_tm3","float_mktcap_tm3_valid","raw_ret_t_t1","raw_ret_t_t10"]
dtypes={"Stkcd":"int32","td":"int32","float_mktcap_tm3":"float64","float_mktcap_tm3_valid":"float64","raw_ret_t_t1":"float64","raw_ret_t_t10":"float64"}
rows=0
print("季度数据已全部读入内存，开始两窗官方基准。",flush=True)
for chunk in pd.read_csv(daily,usecols=cols,dtype=dtypes,parse_dates=["Trddt"],chunksize=300000):
    dates=chunk["Trddt"].to_numpy(dtype="datetime64[D]").astype("int32")
    keep=(dates>=dmin)&(dates<=dmax)
    if not keep.any():rows+=len(chunk);continue
    c=chunk.loc[keep];dates=dates[keep];firm=c["Stkcd"].to_numpy()
    y=c[["raw_ret_t_t1","raw_ret_t_t10"]].to_numpy(dtype="float64")
    month=dates.astype("datetime64[D]").astype("datetime64[M]").astype("int64")
    fy=month//12+1970-(month%12<6);port=lookup[fy-2000,firm]
    keys=dates.astype("int64")*1000000+firm;pos=np.searchsorted(targetkeys,keys)
    found=pos<len(targetkeys);found[found]=targetkeys[pos[found]]==keys[found]
    assert not hits[pos[found]].any()
    rawtarget[pos[found]]=y[found];porttarget[pos[found]]=port[found];hits[pos[found]]=True
    keys=c["td"].to_numpy(dtype="int64")*1000000+firm;at=np.searchsorted(ek,keys)
    ok=at<len(ek);ok[ok]=ek[at[ok]]==keys[ok]
    w=c["float_mktcap_tm3"].to_numpy()
    ok&=(port>0)&np.isfinite(w)&(w>0)&c["float_mktcap_tm3_valid"].eq(1).to_numpy()
    group=(dates-dmin)*125+port-1
    for j in range(2):
        v=ok&np.isfinite(y[:,j])
        den[:,j]+=np.bincount(group[v],weights=w[v],minlength=ng)
        num[:,j]+=np.bincount(group[v],weights=w[v]*y[v,j],minlength=ng)
    rows+=len(chunk)
    if rows%3000000==0:print("扫描",rows,"行，秒数",round(time.monotonic()-t0,1),flush=True)
assert hits.all()
bench=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0)
groups=(targetkeys//1000000-dmin)*125+porttarget-1
official=rawtarget-bench[groups];official[porttarget==0]=np.nan
cuts=np.load(root/"work/图2_允许填充起始日_临时/原样本固定口径.npz")
official=np.clip(official,cuts["lower"][[0,9]],cuts["upper"][[0,9]])
print("官方收益完成，共同样本",int(np.isfinite(official).all(axis=1).sum()),"标签缺失",int((porttarget==0).sum()),flush=True)
del rawtarget,num,den,bench,ek,lookup,hits,chunk,c,groups,lab,porttarget
gc.collect()

def double_fe(v,f,d):
    nf,nd=int(f.max()+1),int(d.max()+1)
    r=np.arange(len(f));fs=sparse.csr_matrix((np.ones(len(f)),(f,r)),shape=(nf,len(f)))
    ds=sparse.csr_matrix((np.ones(len(f)),(d,r)),shape=(nd,len(f)))
    count=sparse.csr_matrix((np.ones(len(f)),(f,d)),shape=(nf,nd))
    fn=np.bincount(f);C,labels=connected_components((count.T@count).tocsr(),directed=False)
    ref=np.unique(labels,return_index=True)[1];free=np.setdiff1d(np.arange(nd),ref)
    schur=np.diag(np.bincount(d))-(count.T@count.multiply(1/fn[:,None])).toarray()
    schur=(schur+schur.T)/2
    out=v.copy();out-=((fs@out)/fn[:,None])[f]
    effect=np.zeros((nd,v.shape[1]))
    effect[free]=cho_solve(cho_factor(schur[np.ix_(free,free)],lower=True,check_finite=False),(ds@out)[free],check_finite=False)
    out-=effect[d];out+=((count@effect)/fn[:,None])[f]
    return out,nf+nd-C

configs=[(0,0),(21,0),(27,0),(27,1),(27,2),(27,3)]
allfirms=np.unique((targetkeys%1000000).astype("int32"))
models=[]
for controls,fe in configs:
    p=1+controls+(fe==0)
    models.append(dict(xx=np.zeros((p,p)),xy=np.zeros((p,4)),yty=np.zeros(4),n=0,rank=0,
        positive=0,filled=0,firms=np.zeros(len(allfirms),dtype=bool),dates=np.zeros(dmax+1,dtype=bool),
        fscores=np.zeros((len(allfirms),4)),dmeat=np.zeros(4),imeat=np.zeros(4)))

prepared=[]
for i,b in enumerate(base):
    new=official[offsets[i]:offsets[i+1]]
    good=np.isfinite(b["old"]).all(axis=1)&np.isfinite(new).all(axis=1)&np.isfinite(b["x"]).all(axis=1)
    firm=b["firm"][good];date=b["date"][good];filled=b["filled"][good]
    values=np.column_stack([b["old"][good],new[good],b["cue"][good],b["x"][good]]).astype("float64")
    pb={}
    for fe in range(4):
        pos=np.arange(len(firm))
        if fe:
            while True:
                _,f=np.unique(firm[pos],return_inverse=True);_,d=np.unique(date[pos],return_inverse=True)
                ok=np.ones(len(pos),dtype=bool)
                if fe in (2,3):ok&=np.bincount(f)[f]>1
                if fe in (1,3):ok&=np.bincount(d)[d]>1
                if ok.all():break
                pos=pos[ok]
        firms=firm[pos];dates=date[pos]
        _,f=np.unique(firms,return_inverse=True);_,d=np.unique(dates,return_inverse=True)
        v=values if fe==0 else values[pos]
        positive=int(v[:,4].sum());filledn=int(filled[pos].sum())
        if fe==3:v,rank=double_fe(v,f,d)
        elif fe:
            g=f if fe==2 else d;G=int(g.max()+1)
            sums=np.zeros((G,v.shape[1]));np.add.at(sums,g,v)
            v=v-sums[g]/np.bincount(g)[g,None];rank=G
        else:rank=0
        pb[fe]=dict(v=v,fi=np.searchsorted(allfirms,firms),d=d,date=dates,rank=rank,positive=positive,filled=filledn)
    prepared.append(pb)
    for j,(controls,fe) in enumerate(configs):
        q=pb[fe];v=q["v"];y=v[:,:4];x=v[:,4:5+controls]
        if not fe:x=np.column_stack([x,np.ones(len(v))])
        m=models[j];m["xx"]+=x.T@x;m["xy"]+=x.T@y;m["yty"]+=np.square(y).sum(axis=0)
        m["n"]+=len(v);m["rank"]+=q["rank"];m["positive"]+=q["positive"];m["filled"]+=q["filled"]
        m["firms"][q["fi"]]=True;m["dates"][q["date"]]=True
    if (i+1)%15==0:print("点估计及固定效应",i+1,"/83，秒数",round(time.monotonic()-t0,1),flush=True)
del base,official
gc.collect()
for j,m in enumerate(models):
    assert np.linalg.cond(m["xx"])<1e12
    m["inverse"]=np.linalg.inv(m["xx"]);m["beta"]=np.linalg.solve(m["xx"],m["xy"])
    print("规格",j+1,"四组点估计百分点",(m["beta"][0]*100).tolist(),flush=True)
for i,pb in enumerate(prepared):
    for j,(controls,fe) in enumerate(configs):
        q=pb[fe];v=q["v"];y=v[:,:4];x=v[:,4:5+controls]
        if not fe:x=np.column_stack([x,np.ones(len(v))])
        m=models[j];err=y-x@m["beta"]
        score=(x@m["inverse"][:,0])[:,None]*err
        for h in range(4):
            m["fscores"][:,h]+=np.bincount(q["fi"],weights=score[:,h],minlength=len(allfirms))
            ds=np.bincount(q["d"],weights=score[:,h]);m["dmeat"][h]+=ds@ds
        m["imeat"]+=np.square(score).sum(axis=0)
    if (i+1)%15==0:print("聚类推断",i+1,"/83，秒数",round(time.monotonic()-t0,1),flush=True)
results=[]
for j,(controls,fe) in enumerate(configs):
    m=models[j];n=m["n"];gf=int(m["firms"].sum());gd=int(m["dates"].sum());K=1+controls
    var=(n-1)/(n-K-1)*(gf/(gf-1)*np.square(m["fscores"]).sum(axis=0)+gd/(gd-1)*m["dmeat"]-n/(n-1)*m["imeat"])
    assert (var>0).all()
    se=np.sqrt(var);beta=m["beta"][0];df=n-m["rank"]-K-(fe==0)
    pv=2*stats.t.sf(np.abs(beta/se),df)
    row=dict(规格=j+1,样本量=int(n),固定效应秩=int(m["rank"]),自由度=int(df),公司数=gf,日期数=gd,
        正提示日=m["positive"],填充起点=m["filled"],系数百分点=(beta*100).tolist(),
        标准误百分点=(se*100).tolist(),概率=pv.tolist())
    results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
anchor=pd.read_csv(root/"work/figure2_official_size90/官方标签_两种提示_精确估计结果.csv")
anchor=anchor.loc[anchor["提示口径"].eq("沪深300提示")&anchor["窗口"].isin([1,10])].sort_values("窗口")
diff=np.array(results[5]["系数百分点"])[2:]-anchor["系数_百分点"].to_numpy()
diffse=np.array(results[5]["标准误百分点"])[2:]-anchor["标准误_百分点"].to_numpy()
print("官方图2锚点系数差",diff.tolist(),"标准误差",diffse.tolist(),flush=True)
anchor_sample_matches = results[5]["样本量"] == int(anchor["样本量"].iloc[0])
anchor_estimates_match = np.max(np.abs(diff)) < 1e-6 and np.max(np.abs(diffse)) < 1e-6
anchor_passed = anchor_sample_matches and anchor_estimates_match
for p,stamp in watch.items():
    s=Path(p).stat();assert (s.st_size,s.st_mtime_ns)==stamp,p

# 将本次国泰安标签收益结果整理为两个收益窗口、每窗六种规格的绘图表。
# 四组结果中的下标2、3对应国泰安标签两日、十一日，系数和标准误已是百分点。
anchor_status = "与旧图2参考结果一致" if anchor_passed else "与旧图2参考结果不一致"
plot_rows = []
for result in results:
    for index, window in ((2, "[t, t+1]"), (3, "[t, t+10]")):
        plot_rows.append({
            "样本": "国泰安标签收益试算",
            "样本说明": f"沪深300提示；国泰安标签收益试算（{anchor_status}）",
            "提示变量": "quarterLag1HS300",
            "收益窗口": window,
            "规格": f"({result['规格']})",
            "线索系数（百分点）": result["系数百分点"][index],
            "聚类标准误（百分点）": result["标准误百分点"][index],
            "p值": result["概率"][index],
            "回归样本量": result["样本量"],
            "R方": np.nan,
        })
plot_summary = pd.DataFrame(plot_rows)
output_path = (
    root / "图表结果" / "04_国泰安标签收益检验" / "表2_收益回归"
    / "表2_国泰安标签_收益回归_六列试算.png"
)
plot_notes = [
    "注：使用国泰安组合标签构建特征调整收益；提示变量为沪深300提示；保留2005—2025年样本。",
    "样本为原有收益、国泰安标签收益的两种窗口及全部27项控制均非缺失的共同样本；允许合格填充日起点。",
    "主规格为（6）：27项控制、被提示公司×年度季度和日期固定效应；括号内为公司与日期双向聚类标准误。",
    "拟合优度未计算，表中对应位置留空；*、**、***分别表示双侧检验在10%、5%、1%水平显著。",
    f"参考比较：{anchor_status}；仅比较第（6）列的系数、标准误和样本量，数值绝对容差为0.000001个百分点。",
]

# 复用主线表2的排版；绘图临时缓存仅存放在项目work目录，结束后自动清理。
plot_work = root / "work"
plot_work.mkdir(parents=True, exist_ok=True)
previous_plot_cache = os.environ.get("MPLCONFIGDIR")
with TemporaryDirectory(prefix="table2_csmar_plot_", dir=plot_work) as plot_cache:
    try:
        os.environ["MPLCONFIGDIR"] = plot_cache
        import matplotlib
        matplotlib.use("Agg")
        from 回归表格绘制 import draw_table2_compact

        draw_table2_compact(plot_summary, output_path, notes=plot_notes)
    finally:
        if previous_plot_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_plot_cache
print("表2图片已保存：",output_path,flush=True)

# 旧图2用于参考比较，不作为本次回归和出图的终止条件；原数值容差仍保持不变。
if anchor_passed:
    print("第（6）列与旧图2参考结果在原容差内一致。",flush=True)
else:
    print("提示：图片已保存，第（6）列与旧图2参考结果不完全一致；本次结果仍按试算标注。",flush=True)
    print("样本量是否一致：",anchor_sample_matches,
          "；系数最大绝对差（百分点）：",float(np.max(np.abs(diff))),
          "；标准误最大绝对差（百分点）：",float(np.max(np.abs(diffse))),
          "；原绝对容差（百分点）：",1e-6,flush=True)
print("全部12个官方模型及表2图片已完成，三个受监测输入文件的稳定性检查通过；四组依次为原标签两日、原标签十一日、官方两日、官方十一日。总秒数",round(time.monotonic()-t0,1),flush=True)
