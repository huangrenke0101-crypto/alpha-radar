import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

st.set_page_config(page_title="A股 Alpha Radar", page_icon="📡", layout="wide")
st.title("📡 A股 Alpha Radar")
st.caption("V1.0 手机/电脑网页研究版｜收盘后短线选股｜不自动下单")

st.sidebar.header("扫描设置")
mode = st.sidebar.radio("数据模式", ["演示数据", "公开行情快照（AKShare）"])
min_cap = st.sidebar.number_input("总市值下限（亿元）", 1, 1000, 50)
max_cap = st.sidebar.number_input("总市值上限（亿元）", 1, 2000, 300)
min_amount = st.sidebar.number_input("最低成交额（万元）", 0, 1000000, 10000, step=1000)
top_n = st.sidebar.slider("展示候选数量", 5, 50, 20)

@st.cache_data(ttl=900)
def get_data(mode):
    if mode == "演示数据":
        rng = np.random.default_rng(17)
        n = 260
        caps = rng.uniform(35, 340, n)
        close = rng.uniform(8, 65, n)
        pct = rng.normal(0.5, 3.0, n)
        amount = rng.lognormal(np.log(2e8), .8, n)
        turnover = rng.uniform(1, 16, n)
        ret3 = rng.normal(2, 6, n)
        ret5 = rng.normal(3, 9, n)
        ret10 = rng.normal(5, 13, n)
        vol_ratio = rng.uniform(.6, 3.5, n)
        bull = rng.random(n) > .45
        breakout = rng.random(n) > .78
        pullback = rng.random(n) > .70
        return pd.DataFrame({
            "代码":[f"DEMO{i+1:04d}" for i in range(n)],
            "名称":[f"演示标的{i+1:03d}" for i in range(n)],
            "总市值(亿)":caps, "收盘价":close, "涨跌幅(%)":pct,
            "成交额(元)":amount, "换手率(%)":turnover, "3日涨幅(%)":ret3,
            "5日涨幅(%)":ret5, "10日涨幅(%)":ret10, "量比":vol_ratio,
            "均线多头":bull, "阶段突破":breakout, "强势回踩":pullback
        }), "合成演示数据，不是真实股票或行情"
    import akshare as ak
    raw = ak.stock_zh_a_spot_em()
    mapping = {"代码":"代码","名称":"名称","最新价":"收盘价","涨跌幅":"涨跌幅(%)",
               "成交额":"成交额(元)","换手率":"换手率(%)","总市值":"总市值(元)"}
    missing = [c for c in mapping if c not in raw.columns]
    if missing:
        raise ValueError(f"行情字段发生变化，缺少：{missing}")
    df = raw[list(mapping)].rename(columns=mapping).copy()
    for c in ["收盘价","涨跌幅(%)","成交额(元)","换手率(%)","总市值(元)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["总市值(亿)"] = df["总市值(元)"]/1e8
    df = df.dropna(subset=["收盘价","总市值(亿)","成交额(元)"])
    # 单日快照不能推算历史动量/均线；保留缺失值，不伪造历史指标
    for c in ["3日涨幅(%)","5日涨幅(%)","10日涨幅(%)","量比"]:
        df[c] = np.nan
    df["均线多头"] = False
    df["阶段突破"] = False
    df["强势回踩"] = False
    return df, "AKShare公开行情单日快照；历史指标未接入"

try:
    with st.spinner("正在读取数据…"):
        df, note = get_data(mode)
except Exception as e:
    st.error(f"数据获取失败：{e}")
    st.info("可切换到演示数据，或稍后重试。公开行情接口可能受网络、限流及第三方接口变化影响。")
    st.stop()

st.info(f"数据说明：{note}。数据日期：{date.today().isoformat()}（本地运行日期）")
if "总市值(亿)" not in df:
    st.error("数据格式不完整。")
    st.stop()

df = df[(df["总市值(亿)"]>=min_cap)&(df["总市值(亿)"]<=max_cap)]
df = df[df["成交额(元)"]>=min_amount*1e4]
df = df[~df["名称"].astype(str).str.contains("ST|退", case=False, na=False)].copy()

def clip(x): return float(np.clip(x,0,100))
def calc(row):
    mom_vals = [row.get("3日涨幅(%)"),row.get("5日涨幅(%)"),row.get("10日涨幅(%)")]
    mom_vals = [v for v in mom_vals if pd.notna(v)]
    mom = clip(50 + np.mean(mom_vals)*2.5) if mom_vals else 50
    vr = row.get("量比")
    vol = clip(35 + min(vr,3)*18) if pd.notna(vr) else 50
    rel = clip(50 + row["涨跌幅(%)"]*2)
    tech = 45 + (25 if row["均线多头"] else 0) + (25 if row["阶段突破"] else 0) + (10 if row["强势回踩"] else 0)
    liq = clip(20 + min(row["成交额(元)"]/1e8,10)*8)
    context = 50
    score = mom*.25 + vol*.20 + rel*.15 + clip(tech)*.10 + liq*.10 + context*.20
    return pd.Series([mom,vol,rel,clip(tech),liq,context,score])
if len(df):
    df[["动量分","量价分","相对强度分","技术形态分","流动性分","市场/事件分","Alpha分"]] = df.apply(calc,axis=1)
    df = df.sort_values(["Alpha分","成交额(元)"],ascending=False)
    df["等级"] = np.select([df["Alpha分"]>=85,df["Alpha分"]>=75,df["Alpha分"]>=65],["S","A","B"],default="C")
    df["策略标签"] = np.where(df["阶段突破"],"突破候选",np.where(df["强势回踩"],"回踩候选","动量/观察"))
    df["风险提示"] = np.where((df["涨跌幅(%)"]>=9.5)|(df["换手率(%)"]>=25),"高波动风险","需核实公告及承接")
else:
    st.warning("当前条件下没有符合筛选条件的股票。")
    st.stop()

adv = int((df["涨跌幅(%)"]>0).sum())
dec = int((df["涨跌幅(%)"]<0).sum())
c1,c2,c3,c4 = st.columns(4)
c1.metric("筛选后股票数",f"{len(df):,}")
c2.metric("上涨家数",adv)
c3.metric("下跌家数",dec)
c4.metric("平均涨跌幅",f"{df['涨跌幅(%)'].mean():.2f}%")

st.subheader("Alpha 候选排行榜")
cols = ["代码","名称","总市值(亿)","收盘价","涨跌幅(%)","成交额(元)","换手率(%)","Alpha分","等级","策略标签","风险提示"]
st.dataframe(df[cols].head(top_n),use_container_width=True,hide_index=True)

st.subheader("个股评分拆解")
choice = st.selectbox("选择股票",df.head(top_n)["代码"]+" "+df.head(top_n)["名称"])
code = choice.split()[0]
r = df[df["代码"]==code].iloc[0]
st.write(f"**{r['名称']}｜{code}｜{r['Alpha分']:.1f}分｜{r['等级']}级**")
st.write(f"策略标签：{r['策略标签']}。风险提示：{r['风险提示']}。")
parts = pd.DataFrame({"维度":["动量","量价","相对强度","技术形态","流动性","市场/事件"],
                      "得分":[r["动量分"],r["量价分"],r["相对强度分"],r["技术形态分"],r["流动性分"],r["市场/事件分"]]})
st.bar_chart(parts.set_index("维度"))
st.warning("交易纪律：收盘后评分仅用于建立观察名单。次日需重新确认板块强度、价格承接、成交量及公告风险；不追高，不以模型分数直接下单。")

st.subheader("数据与模型边界")
st.markdown("""
- 演示模式使用合成数据，绝不能当作真实股票推荐。
- AKShare模式目前只读取单日公开行情快照，无法提供真实的3/5/10日动量、历史均线、量比和突破判断。
- 当前缺少历史行情数据库、板块历史数据、点时回测及完整交易成本模拟，因此尚不是可用于实盘决策的成熟策略。
- Alpha分是规则模型的相对排序，不代表上涨概率或收益保证。
""")
st.caption("A股 Alpha Radar V1.0｜研究与教育用途，不自动下单。")
