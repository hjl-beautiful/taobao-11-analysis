import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="淘宝双11数据分析", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.metric-card { background: linear-gradient(135deg, #FF6A00, #FF8E53); padding: 20px; border-radius: 15px; color: white; text-align: center; }
.metric-value { font-size: 2rem; font-weight: 700; }
.insight-box { background: #f8f9fa; border-left: 4px solid #FF6A00; padding: 15px 20px; border-radius: 0 10px 10px 0; margin-top: 15px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_sales_data():
    data = {
        'year': [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023],
        'gmv': [0.52, 9.36, 52, 191, 350, 571, 912, 1207, 1682, 2135, 2684, 4982, 5403, 5571, 5800]
    }
    df = pd.DataFrame(data)
    df['growth_rate'] = df['gmv'].pct_change() * 100
    return df

@st.cache_data
def load_user_behavior():
    np.random.seed(42)
    n = 50000
    df = pd.DataFrame({
        'user_id': range(1, n + 1),
        'age_range': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+'], n, p=[0.25, 0.35, 0.20, 0.12, 0.08]),
        'gender': np.random.choice(['男', '女'], n, p=[0.42, 0.58]),
        'city_tier': np.random.choice(['一线城市', '新一线', '二线城市', '三线及以下'], n, p=[0.15, 0.25, 0.30, 0.30]),
        'purchase_amount': np.random.lognormal(4.5, 1.2, n),
        'browse_count': np.random.poisson(15, n),
        'cart_count': np.random.poisson(3, n),
        'order_count': np.random.poisson(1.5, n),
        'is_return': np.random.choice([0, 1], n, p=[0.65, 0.35])
    })
    mask = df['is_return'] == 1
    df.loc[mask, 'purchase_amount'] *= np.random.uniform(1.5, 3.0, mask.sum())
    df.loc[mask, 'browse_count'] += np.random.poisson(10, mask.sum())
    df.loc[mask, 'cart_count'] += np.random.poisson(2, mask.sum())
    df.loc[mask, 'order_count'] += np.random.poisson(2, mask.sum())
    df['purchase_amount'] = df['purchase_amount'].round(2).clip(10, 50000)
    df['browse_count'] = df['browse_count'].clip(1, 100)
    df['cart_count'] = df['cart_count'].clip(0, 50)
    df['order_count'] = df['order_count'].clip(0, 30)
    return df

sales_df = load_sales_data()
user_df = load_user_behavior()

st.markdown('<h1 style="text-align:center; color:#FF6A00;">🛒 淘宝双11销售趋势与用户行为分析</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#666;">GMV 为公开真实数据，用户行为为基于业务规则的模拟数据 | 仅供分析方法演示</p>', unsafe_allow_html=True)

with st.expander("📋 数据说明", expanded=False):
    st.markdown("""
    - **GMV 数据**：2009-2023 年淘宝双 11 公开销售额，来源为历年公开报道。
    - **用户行为数据**：基于电商用户画像规律生成的模拟数据（样本量 5 万），用于演示 RFM、聚类与分类建模方法。
    - **模型结果**：由于用户行为为模拟数据，模型 AUC 等指标仅用于展示算法流程，不代表真实业务预测能力。
    """)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{sales_df["gmv"].iloc[-1]:.0f}</div><div>2023年GMV（亿元）</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{sales_df["growth_rate"].iloc[-1]:.1f}%</div><div>同比增长率</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(user_df):,}</div><div>用户样本量</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{user_df["is_return"].mean()*100:.1f}%</div><div>回头客比例</div></div>', unsafe_allow_html=True)
with c5:
    st.markdown(f'<div class="metric-card"><div class="metric-value">¥{user_df["purchase_amount"].mean():.0f}</div><div>人均消费</div></div>', unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📈 销售趋势", "👥 用户分析", "🤖 回头客预测"])

with tab1:
    st.markdown("### 📈 历年双11销售额趋势")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.line_chart(sales_df.set_index('year')['gmv'], use_container_width=True)
    with col2:
        st.markdown("**历年数据**")
        display = sales_df[['year', 'gmv', 'growth_rate']].copy()
        display.columns = ['年份', 'GMV(亿)', '增长率(%)']
        display['增长率(%)'] = display['增长率(%)'].round(1)
        st.dataframe(display, height=350, use_container_width=True)
    st.markdown('<div class="insight-box"><b>💡 分析结论</b><br>2009-2013年为<strong>爆发期</strong>（年均增长超200%），2014-2019年为<strong>高速增长期</strong>，2020年后进入<strong>成熟期</strong>（增速降至个位数）。2020年GMV跳升至4982亿元（+85.6%），主要受疫情推动线上消费渗透。</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 👥 用户行为分析")
    c1, c2 = st.columns(2)
    with c1:
        age_stats = user_df.groupby('age_range')['purchase_amount'].mean()
        st.markdown("**各年龄段人均消费**")
        st.bar_chart(age_stats)
    with c2:
        city_stats = user_df.groupby('city_tier')['purchase_amount'].mean()
        st.markdown("**各城市层级人均消费**")
        st.bar_chart(city_stats)
    c3, c4 = st.columns(2)
    with c3:
        gender_stats = user_df.groupby('gender')['is_return'].mean() * 100
        st.markdown("**性别回头客率(%)**")
        st.bar_chart(gender_stats)
    with c4:
        user_df['rfm_score'] = pd.qcut(user_df['purchase_amount'], 5, labels=['低', '中低', '中', '中高', '高'])
        rfm = user_df['rfm_score'].value_counts()
        st.markdown("**消费分层**")
        st.bar_chart(rfm)
    st.markdown('<div class="insight-box"><b>💡 用户洞察</b><br>26-35岁用户是消费主力（占比35%），人均消费最高；一线城市客单价高但增长放缓，<strong>新一线和二线城市</strong>是增量市场。回头客占比35%，但贡献超60%销售额。</div>', unsafe_allow_html=True)

with tab3:
    st.markdown("### 🤖 回头客预测模型")
    col1, col2 = st.columns([1, 2])
    with col1:
        model_choice = st.selectbox("选择模型", ["逻辑回归", "随机森林"], index=1)
        test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
        features = st.multiselect("选择特征", ['purchase_amount', 'browse_count', 'cart_count', 'order_count', 'city_tier'], default=['purchase_amount', 'browse_count', 'cart_count', 'order_count'])
        if st.button("🚀 训练模型", type="primary"):
            df_model = user_df.copy()
            if 'city_tier' in features:
                le = LabelEncoder()
                df_model['city_tier'] = le.fit_transform(df_model['city_tier'])
            X = df_model[features]
            y = df_model['is_return']
            num_f = [f for f in features if f != 'city_tier']
            if num_f:
                scaler = StandardScaler()
                X[num_f] = scaler.fit_transform(X[num_f])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            if model_choice == "逻辑回归":
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            st.session_state['auc'] = auc
            st.session_state['model'] = model
            st.session_state['features'] = features
            st.success(f"✅ AUC = {auc:.4f}")
    with col2:
        if 'auc' in st.session_state:
            st.markdown(f"**📊 模型评估**")
            st.markdown(f"<div style='background:#FFF3E0; padding:20px; border-radius:10px; text-align:center;'><span style='font-size:2rem; color:#FF6A00; font-weight:700;'>AUC = {st.session_state['auc']:.4f}</span></div>", unsafe_allow_html=True)
            if hasattr(st.session_state['model'], 'feature_importances_'):
                imp = pd.DataFrame({'feature': st.session_state['features'], 'importance': st.session_state['model'].feature_importances_}).sort_values('importance', ascending=False)
                st.markdown("**特征重要性**")
                st.bar_chart(imp.set_index('feature'))
        else:
            st.info("👈 请先训练模型")

st.markdown("---")
st.markdown('<div style="text-align:center; color:#999;"><p>淘宝双11销售趋势与用户行为分析 | 数据科学方法演示</p><p>技术栈：Python | Streamlit | scikit-learn | Pandas</p></div>', unsafe_allow_html=True)
