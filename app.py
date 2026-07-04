# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import html

st.set_page_config(page_title="Retail Intelligence Suite", page_icon="🛍️", layout="wide")

#  NEW FIXED CODE
st.markdown("""
<style>
    .metric-card {background: white; border: 1px solid #e2e8f0; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);}
    .recommendation-card {background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #6366f1; padding: 18px; border-radius: 8px; min-height: 140px;}
</style>
""", unsafe_allow_html=True) # <-- Fixed parameter name here!

@st.cache_data
def load_production_data():
    rfm = pd.read_csv("rfm_segmented.csv")
    products = pd.read_csv("product_list.csv")['Description'].dropna().unique().tolist()
    products.sort()
    with open("item_similarity.pkl", "rb") as f:
        similarity_data = pickle.load(f)
    
    # Try opening zipped or raw dataset variants
    df_raw = None
    for f_variant in ["online_retail_compressed.csv.gz", "online_retail.csv"]:
        try:
            df_raw = pd.read_csv(f_variant)
            df_raw['InvoiceDate'] = pd.to_datetime(df_raw['InvoiceDate'])
            df_raw['TotalAmount'] = df_raw['Quantity'] * df_raw['UnitPrice']
            break
        except Exception:
            continue
    return rfm, products, similarity_data, df_raw

try:
    rfm_df, product_list, similarity_matrix, raw_df = load_production_data()
except FileNotFoundError:
    st.error("🚨 Deployment files are missing! Run `train_pipeline.py` first.")
    st.stop()

st.title("🛍️ Advanced E-Commerce Intelligence Suite")

# --- FIXED METRICS BANNER ---
m1, m2, m3, m4 = st.columns(4)
with m1: 
    st.markdown(f'<div class="metric-card"><span style="color:#64748b; font-size:13px; font-weight:600;">ACTIVE CUSTOMERS</span><h2 style="margin:5px 0 0 0; color:#0f172a;">{len(rfm_df):,}</h2></div>', unsafe_allow_html=True)
with m2: 
    st.markdown(f'<div class="metric-card"><span style="color:#64748b; font-size:13px; font-weight:600;">AVG RECENCY</span><h2 style="margin:5px 0 0 0; color:#0f172a;">{int(rfm_df["Recency"].mean())} Days</h2></div>', unsafe_allow_html=True)
with m3: 
    st.markdown(f'<div class="metric-card"><span style="color:#64748b; font-size:13px; font-weight:600;">AVG FREQUENCY</span><h2 style="margin:5px 0 0 0; color:#0f172a;">{rfm_df["Frequency"].mean():.1f} Orders</h2></div>', unsafe_allow_html=True)
with m4: 
    st.markdown(f'<div class="metric-card"><span style="color:#64748b; font-size:13px; font-weight:600;">AVG LIFETIME SPEND</span><h2 style="margin:5px 0 0 0; color:#0f172a;">${rfm_df["Monetary"].mean():,.2f}</h2></div>', unsafe_allow_html=True)

st.write("")
tab_recommend, tab_segments, tab_eda = st.tabs(["🎯 Recommendation Engine", "👥 Customer Segments", "📊 Business Dashboard"])

# TAB 1: RECOMMENDATIONS
with tab_recommend:
    st.header("🎯 Product Recommendation Module")
    selected_product = st.selectbox("Search or Select a Base Inventory Item:", options=[""] + product_list)
    
    if selected_product:
        is_dict = isinstance(similarity_matrix, dict)
        if (selected_product in similarity_matrix) if is_dict else (selected_product in similarity_matrix.index):
            if is_dict:
                raw_recs = similarity_matrix[selected_product][:5]
                recommendations = pd.Series({k: v for k, v in raw_recs})
            else:
                recommendations = similarity_matrix[selected_product].sort_values(ascending=False).iloc[1:6]
                
            # --- FIXED RECOMMENDATION CARDS LOOP IN TAB 1 ---
            st.write(f"### ✨ Top 5 Product Recommendations for **'{selected_product}'**")
            cols = st.columns(5)
            for idx, (prod_name, score) in enumerate(recommendations.items()):
                safe_name = html.escape(prod_name)
                with cols[idx]:
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <span style="font-weight: bold; color: #4f46e5; font-size: 13px;">RANK #{idx+1}</span>
                        <p style="font-size:14px; margin:8px 0; font-weight:500; min-height:55px;">{safe_name}</p>
                        <div style="background:#f0fdf4; border-radius:4px; padding:3px; text-align:center;">
                            <span style="font-size:12px; color:#15803d; font-weight:bold;">Match: {score:.2%}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # --- THIS SECTION IS NOW CLEANLY ALIGNED ---
            st.subheader("🔗 Correlation Proximity Heatmap")
            sub_items = [selected_product] + list(recommendations.index)
            if is_dict:
                m_data = np.zeros((len(sub_items), len(sub_items)))
                for i, item_i in enumerate(sub_items):
                    item_i_matches = dict(similarity_matrix.get(item_i, []))
                    for j, item_j in enumerate(sub_items):
                        m_data[i, j] = 1.0 if i == j else item_i_matches.get(item_j, 0.0)
                sub_matrix = pd.DataFrame(m_data, index=sub_items, columns=sub_items)
            else:
                sub_matrix = similarity_matrix.loc[sub_items, sub_items]
                
            labels = [html.escape(n[:25]+'...') if len(n)>25 else html.escape(n) for n in sub_items]
            fig_h = px.imshow(sub_matrix.values, x=labels, y=labels, color_continuous_scale="Viridis")
            st.plotly_chart(fig_h, use_container_width=True)
# TAB 2: SEGMENTS
with tab_segments:
    st.header("👥 Customer Clusters")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig_3d = px.scatter_3d(rfm_df, x='Recency', y='Frequency', z='Monetary', color='Segment',
                               color_discrete_map={'High-Value':'#10b981', 'Regular':'#3b82f6', 'Occasional':'#f59e0b', 'At-Risk':'#ef4444'},
                               opacity=0.6, log_z=True, title="3D Cluster Map")
        st.plotly_chart(fig_3d, use_container_width=True)
    with c2:
        seg_summary = rfm_df.groupby('Segment').agg(Count=('CustomerID','count'), Spend=('Monetary','mean')).reset_index()
        st.dataframe(seg_summary.style.format({'Spend': '${:,.2f}', 'Count': '{:,}'}), hide_index=True)

# TAB 3: BUSINESS BI (EDA)
with tab_eda:
    st.header("📊 Performance Dashboard")
    if raw_df is not None:
        l, r = st.columns(2)
        with l:
            raw_df['YM'] = raw_df['InvoiceDate'].dt.to_period('M').astype(str)
            m_rev = raw_df.groupby('YM')['TotalAmount'].sum().reset_index()
            st.plotly_chart(px.line(m_rev, x='YM', y='TotalAmount', title="Monthly Revenue Streams"), use_container_width=True)
        with r:
            top_p = raw_df.groupby('Description')['Quantity'].sum().reset_index().sort_values('Quantity', ascending=False).head(10)
            fig_p = px.bar(top_p, x='Quantity', y='Description', orientation='h', title="Top 10 Inventory Items")
            fig_p.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_p, use_container_width=True)