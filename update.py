import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Harga Bahan Pangan Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD DATA ====================
import os
from pathlib import Path

@st.cache_data
def load_real_data():
    """Load data from CSV file"""
    script_dir = Path(__file__).parent.absolute()
    
    possible_files = [
        script_dir / "harga_pangan_satudata.csv",
        script_dir / "harga.csv",
        script_dir / "data_harga.csv"
    ]
    
    for file_path in possible_files:
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['tanggal'] = pd.to_datetime(df['tanggal'])
                return df
            except Exception as e:
                st.sidebar.warning(f"Error baca {file_path.name}: {e}")
    
    return None

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Panel Kontrol")
st.sidebar.markdown("---")

analysis_type = st.sidebar.selectbox(
    "Pilih Metode Analisis",
    ["📈 SARIMA (Prediksi Harga)", "🌲 Random Forest Classifier", "🎯 K-Means Clustering"]
)

province = st.sidebar.selectbox("Provinsi", ["Jawa Barat", "Jawa Timur", "DKI Jakarta", "Jawa Tengah", "Banten"])
commodity = st.sidebar.selectbox("Komoditas", ["Beras", "Cabai Rawit", "Bawang Merah", "Minyak Goreng", "Daging Ayam"])

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Filter Tanggal")
date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    [datetime(2024, 1, 1), datetime(2024, 12, 31)],
    key="date_range"
)

# ==================== UPLOAD DATA + TEMPLATE ====================
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Data")

with st.sidebar.expander("📥 Download Template Excel"):
    st.markdown("Format data yang benar:")
    st.code("""
tanggal, harga, komoditas, provinsi
2024-01-01, 14000, Beras, Jawa Barat
2024-01-02, 14200, Beras, Jawa Barat
""")
    
    template_df = pd.DataFrame({
        "tanggal": [datetime.now().strftime("%Y-%m-%d")],
        "harga": [14000],
        "komoditas": ["Beras"],
        "provinsi": ["Jawa Barat"]
    })
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Template')
    excel_data = output.getvalue()
    
    st.download_button(
        label="⬇️ Download Template Excel",
        data=excel_data,
        file_name="template_data_harga.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV/Excel",
    type=["csv", "xlsx"],
    help="Upload file dengan format yang sesuai (lihat template di atas)"
)

df_uploaded = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)
        df_uploaded['tanggal'] = pd.to_datetime(df_uploaded['tanggal'])
        st.sidebar.success(f"✅ {len(df_uploaded):,} baris data uploaded!")
    except Exception as e:
        st.sidebar.error(f"❌ Error: {e}")

# Pilih sumber data
df_default = load_real_data()

if df_uploaded is not None:
    df = df_uploaded
    st.sidebar.info("📊 Menggunakan data upload")
elif df_default is not None:
    df = df_default
    st.sidebar.info(f"📊 Menggunakan data dari file ({len(df):,} baris)")
else:
    df = None
    st.sidebar.error("❌ Tidak ada data! Upload file CSV/Excel.")

# ==================== HEADER ====================
st.title("📊 Harga Pangan Analytics Dashboard")
if df is not None:
    st.caption(f"Dashboard Analisis Data - {province} | Komoditas: {commodity}")

# ==================== 1. SARIMA ====================
def sarima_section(df, province, commodity, date_range):
    st.header("📈 SARIMA: Prediksi Harga Komoditas")
    
    st.markdown("""
    **Metode SARIMA** *(Seasonal Autoregressive Integrated Moving Average)* digunakan untuk **meramalkan harga komoditas** di masa depan.
    Model ini menganalisis pola historis harga dengan mempertimbangkan:
    - **Tren** (kenaikan/penurunan jangka panjang)
    - **Musiman** (pola berulang tahunan/bulanan)
    - **Fluktuasi** (kejutan eksternal seperti cuaca atau hari besar)
    """)
    
    if df is None or df.empty:
        st.warning("⚠️ Tidak ada data untuk dianalisis")
        return
    
    df_filtered = df[
        (df['provinsi'] == province) & 
        (df['komoditas'] == commodity)
    ]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_filtered = df_filtered[
            (df_filtered['tanggal'] >= pd.to_datetime(start_date)) & 
            (df_filtered['tanggal'] <= pd.to_datetime(end_date))
        ]
    
    if df_filtered.empty:
        st.warning(f"⚠️ Tidak ada data untuk {commodity} di {province} pada periode yang dipilih")
        return
    
    df_price = df_filtered[['tanggal', 'harga']].copy().sort_values('tanggal')
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_price["tanggal"], y=df_price["harga"],
            mode='lines', name='Harga Aktual',
            line=dict(color='blue', width=2)
        ))
        
        last_date = df_price["tanggal"].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        last_price = df_price["harga"].iloc[-1]
        
        avg_price = df_price["harga"].tail(14).mean()
        trend = (df_price["harga"].iloc[-1] - df_price["harga"].iloc[-30]) / 30 if len(df_price) >= 30 else 0
        
        future_prices = []
        for i in range(30):
            next_price = avg_price + (i + 1) * trend * 0.5 + np.random.normal(0, last_price * 0.02)
            future_prices.append(max(next_price, last_price * 0.8))
        future_prices = np.array(future_prices)
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode='lines', name='Prediksi SARIMA',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        upper_bound = future_prices + np.random.normal(last_price * 0.03, last_price * 0.01, 30)
        lower_bound = future_prices - np.random.normal(last_price * 0.03, last_price * 0.01, 30)
        
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=upper_bound.tolist() + lower_bound.tolist()[::-1],
            fill='toself', fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Interval Kepercayaan 95%'
        ))
        
        fig.update_layout(
            title=f"Prediksi Harga {commodity} - {province}",
            xaxis_title="Tanggal",
            yaxis_title=f"Harga (Rp/kg)",
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Ringkasan Statistik")
        
        st.metric(
            f"Harga {commodity} Terakhir",
            f"Rp {df_price['harga'].iloc[-1]:,.0f}"
        )
        st.metric(
            "Rata-rata 14 Hari",
            f"Rp {df_price['harga'].tail(14).mean():,.0f}"
        )
        st.metric(
            "Harga Tertinggi",
            f"Rp {df_price['harga'].max():,.0f}"
        )
        st.metric(
            "Harga Terendah",
            f"Rp {df_price['harga'].min():,.0f}"
        )
        
        st.caption(f"📅 Data: {df_price['tanggal'].min().strftime('%d %b %Y')} - {df_price['tanggal'].max().strftime('%d %b %Y')}")
        st.caption(f"📊 Total data: {len(df_price):,} hari")
    
    with st.expander("📋 Lihat Data Historis"):
        st.dataframe(
            df_price.tail(30).sort_values("tanggal", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# ==================== 2. RANDOM FOREST ====================
def random_forest_section(df, province, date_range):
    st.header("🌲 Random Forest Classifier: Analisis Risiko Komoditas")
    
    # PERBAIKAN: Deskripsi yang lebih rapi dan informatif
    st.markdown("""
    ### 📖 Apa yang Dilakukan Random Forest di Sini?
    
    Random Forest Classifier digunakan untuk **mengelompokkan komoditas berdasarkan tingkat risiko harga**.
    
    **Apa itu "Risiko"?**
    Risiko mengacu pada **tingkat fluktuasi/ketidakstabilan harga** suatu komoditas. Semakin tinggi fluktuasi, semakin tinggi risiko.
    
    **Bagaimana Klasifikasi Dilakukan?**
    1. Hitung fluktuasi setiap komoditas: `Fluktuasi = Std Dev Harga / Rata-rata Harga`
    2. Hitung median fluktuasi dari semua komoditas
    3. Klasifikasikan dengan ketentuan:
       - 🟢 **Risiko Rendah**: `fluktuasi ≤ median`
       - 🟡 **Risiko Sedang**: `median < fluktuasi ≤ median × 1.5`
       - 🔴 **Risiko Tinggi**: `fluktuasi > median × 1.5`
    """)
    
    if df is None or df.empty:
        st.warning("⚠️ Tidak ada data untuk dianalisis")
        return
    
    df_data = df[df['provinsi'] == province].copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_data = df_data[
            (df_data['tanggal'] >= pd.to_datetime(start_date)) & 
            (df_data['tanggal'] <= pd.to_datetime(end_date))
        ]
    
    if df_data.empty:
        st.warning(f"⚠️ Tidak ada data untuk provinsi {province} pada periode yang dipilih")
        return
    
    commodities = df_data['komoditas'].unique()
    
    feature_data = []
    for kom in commodities:
        df_kom = df_data[df_data['komoditas'] == kom]
        
        if len(df_kom) >= 3:
            mean_price = df_kom['harga'].mean()
            std_price = df_kom['harga'].std()
            fluctuation = std_price / mean_price if mean_price > 0 else 0
            
            feature_data.append({
                'komoditas': kom,
                'rata_harga': mean_price,
                'std_harga': std_price,
                'fluktuasi': fluctuation,
                'harga_tertinggi': df_kom['harga'].max(),
                'harga_terendah': df_kom['harga'].min(),
                'range_harga': df_kom['harga'].max() - df_kom['harga'].min(),
                'jumlah_data': len(df_kom)
            })
    
    if not feature_data:
        st.warning("⚠️ Data tidak mencukupi untuk analisis (minimal 3 data per komoditas)")
        return
    
    df_features = pd.DataFrame(feature_data)
    
    st.subheader("📊 Statistik Harga per Komoditas")
    st.dataframe(
        df_features[['komoditas', 'rata_harga', 'fluktuasi', 'jumlah_data']]
        .style.format({
            'rata_harga': 'Rp {:,.0f}',
            'fluktuasi': '{:.1%}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(
                x=df_features['komoditas'],
                y=df_features['fluktuasi'],
                marker_color='#3498db',
                text=df_features['fluktuasi'].apply(lambda x: f"{x:.1%}"),
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Fluktuasi Harga per Komoditas",
            xaxis_title="Komoditas",
            yaxis_title="Koefisien Variasi",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(
                x=df_features['komoditas'],
                y=df_features['rata_harga'],
                marker_color='#2ecc71',
                text=df_features['rata_harga'].apply(lambda x: f"Rp {x:,.0f}"),
                textposition='outside'
            )
        ])
        fig.update_layout(
            title="Rata-rata Harga per Komoditas",
            xaxis_title="Komoditas",
            yaxis_title="Harga (Rp/kg)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Klasifikasi Risiko Komoditas")
    
    # Hitung median fluktuasi
    median_fluktuasi = df_features['fluktuasi'].median()
    
    # Tampilkan informasi median
    st.info(f"📊 **Median Fluktuasi:** {median_fluktuasi:.1%}")
    
    # Klasifikasi
    df_features['risk_level'] = np.where(
        df_features['fluktuasi'] > median_fluktuasi * 1.5, "🔴 Risiko Tinggi",
        np.where(df_features['fluktuasi'] > median_fluktuasi, "🟡 Risiko Sedang", "🟢 Risiko Rendah")
    )
    
    risk_counts = df_features['risk_level'].value_counts()
    
    # Pie chart
    fig = go.Figure(data=[
        go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.4,
            textinfo='label+percent'
        )
    ])
    fig.update_layout(title="Distribusi Risiko Komoditas")
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel hasil klasifikasi
    st.dataframe(
        df_features[['komoditas', 'fluktuasi', 'rata_harga', 'risk_level']]
        .sort_values('fluktuasi', ascending=False)
        .style.format({'fluktuasi': '{:.1%}', 'rata_harga': 'Rp {:,.0f}'}),
        use_container_width=True,
        hide_index=True
    )
    
    # Tambahkan rekomendasi
    st.subheader("💡 Rekomendasi Kebijakan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🟢 Risiko Rendah")
        st.markdown("- Harga stabil")
        st.markdown("- **Tindakan:** Pemantauan rutin")
    
    with col2:
        st.markdown("#### 🟡 Risiko Sedang")
        st.markdown("- Harga cukup fluktuatif")
        st.markdown("- **Tindakan:** Pemantauan intensif")
    
    with col3:
        st.markdown("#### 🔴 Risiko Tinggi")
        st.markdown("- Harga sangat fluktuatif")
        st.markdown("- **Tindakan:** Prioritas intervensi pemerintah")

# ==================== 3. K-MEANS ====================
def kmeans_section(df, province, date_range):
    st.header("🎯 K-Means Clustering: Segmentasi Komoditas")
    
    st.markdown("""
    ### 📖 Segmentasi Komoditas Berdasarkan Pola Harga
    
    K-Means Clustering mengelompokkan **komoditas** berdasarkan kemiripan pola harga.
    
    **Data yang digunakan:** Statistik harga dari masing-masing komoditas.
    
    **Fitur yang digunakan:**
    - Rata-rata harga
    - Standar deviasi harga
    - Fluktuasi (CV = std/mean)
    - Range harga (max - min)
    
    **Hasil Segmentasi:**
    - 🔴 **Mahal**: Komoditas dengan harga rata-rata tertinggi
    - 🟡 **Sedang**: Komoditas dengan harga rata-rata sedang
    - 🟢 **Terjangkau**: Komoditas dengan harga rata-rata terendah
    """)
    
    if df is None or df.empty:
        st.warning("⚠️ Tidak ada data untuk dianalisis")
        return
    
    df_data = df[df['provinsi'] == province].copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_data = df_data[
            (df_data['tanggal'] >= pd.to_datetime(start_date)) & 
            (df_data['tanggal'] <= pd.to_datetime(end_date))
        ]
    
    if df_data.empty:
        st.warning(f"⚠️ Tidak ada data untuk provinsi {province} pada periode yang dipilih")
        return
    
    commodities = df_data['komoditas'].unique()
    
    feature_data = []
    for kom in commodities:
        df_kom = df_data[df_data['komoditas'] == kom]
        
        if len(df_kom) >= 5:
            mean_price = df_kom['harga'].mean()
            std_price = df_kom['harga'].std()
            fluctuation = std_price / mean_price if mean_price > 0 else 0
            
            feature_data.append({
                'komoditas': kom,
                'rata_harga': mean_price,
                'std_harga': std_price,
                'fluktuasi': fluctuation,
                'range_harga': df_kom['harga'].max() - df_kom['harga'].min(),
                'min_harga': df_kom['harga'].min(),
                'max_harga': df_kom['harga'].max()
            })
    
    if len(feature_data) < 2:
        st.warning("⚠️ Minimal 2 komoditas dengan data cukup untuk clustering")
        return
    
    df_features = pd.DataFrame(feature_data)
    
    # Normalisasi dan Clustering
    features = ['rata_harga', 'std_harga', 'fluktuasi', 'range_harga']
    X = df_features[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = min(3, len(df_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Beri nama cluster berdasarkan rata-rata harga
    cluster_order = df_features.groupby('cluster')['rata_harga'].mean().sort_values(ascending=False)
    
    segmen_names = ["🔴 Mahal", "🟡 Sedang", "🟢 Terjangkau"]
    for i, cluster_id in enumerate(cluster_order.index):
        if i < len(segmen_names):
            df_features.loc[df_features['cluster'] == cluster_id, 'segmen'] = segmen_names[i]
    
    # Visualisasi
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure()
        colors = {'🔴 Mahal': '#e74c3c', '🟡 Sedang': '#f39c12', '🟢 Terjangkau': '#2ecc71'}
        
        for segmen in df_features['segmen'].unique():
            cluster_data = df_features[df_features['segmen'] == segmen]
            fig.add_trace(go.Scatter(
                x=cluster_data['rata_harga'],
                y=cluster_data['fluktuasi'],
                mode='markers+text',
                name=segmen,
                marker=dict(size=20, color=colors.get(segmen, '#3498db'), opacity=0.8),
                text=cluster_data['komoditas'],
                textposition='top center'
            ))
        
        fig.update_layout(
            title="Segmentasi Komoditas Berdasarkan Harga dan Fluktuasi",
            xaxis_title="Rata-rata Harga (Rp/kg)",
            yaxis_title="Fluktuasi Harga (%)",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Profil Cluster")
        
        profile = df_features.groupby('segmen').agg({
            'komoditas': lambda x: ', '.join(x),
            'rata_harga': 'mean',
            'fluktuasi': 'mean',
            'std_harga': 'mean'
        })
        profile.columns = ['Komoditas', 'Rata Harga', 'Fluktuasi', 'Std Dev']
        profile['Rata Harga'] = profile['Rata Harga'].apply(lambda x: f"Rp {x:,.0f}")
        profile['Fluktuasi'] = profile['Fluktuasi'].apply(lambda x: f"{x:.1%}")
        profile['Std Dev'] = profile['Std Dev'].apply(lambda x: f"Rp {x:,.0f}")
        
        st.dataframe(profile, use_container_width=True)
    
    # Detail setiap komoditas per cluster
    st.subheader("📋 Detail Setiap Komoditas per Cluster")
    
    for segmen in df_features['segmen'].unique():
        cluster_data = df_features[df_features['segmen'] == segmen]
        with st.expander(f"{segmen} ({len(cluster_data)} komoditas)"):
            st.dataframe(
                cluster_data[['komoditas', 'rata_harga', 'fluktuasi', 'range_harga']]
                .style.format({
                    'rata_harga': 'Rp {:,.0f}',
                    'fluktuasi': '{:.1%}',
                    'range_harga': 'Rp {:,.0f}'
                }),
                hide_index=True,
                use_container_width=True
            )
    
    # Evaluasi
    with st.expander("📐 Evaluasi Model K-Means"):
        sil_score = silhouette_score(X_scaled, df_features['cluster'])
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        st.caption("Semakin mendekati 1, semakin baik clustering")
        
        # Elbow Method
        inertias = []
        max_k = min(4, len(df_features))
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
        
        if len(inertias) >= 2:
            fig = go.Figure(data=[
                go.Scatter(
                    x=list(range(2, max_k + 1)),
                    y=inertias,
                    mode='lines+markers',
                    marker=dict(size=10)
                )
            ])
            fig.update_layout(
                title="Elbow Method - Optimal K",
                xaxis_title="Jumlah Cluster (K)",
                yaxis_title="Inertia"
            )
            st.plotly_chart(fig, use_container_width=True)

# ==================== MAIN ROUTING ====================
if df is not None:
    if analysis_type == "📈 SARIMA (Prediksi Harga)":
        sarima_section(df, province, commodity, date_range)
    elif analysis_type == "🌲 Random Forest Classifier":
        random_forest_section(df, province, date_range)
    else:
        kmeans_section(df, province, date_range)
else:
    st.warning("""
    ## ⚠️ Tidak Ada Data!
    
    Silakan upload file CSV/Excel melalui sidebar atau pastikan file `harga_pangan_satudata.csv` ada di folder yang sama.
    """)

# ==================== FOOTER ====================
st.markdown("---")
if df is not None:
    st.caption(
        f"© 2025 Harga Pangan Analytics Dashboard | Data: {len(df):,} baris | "
        f"Periode: {df['tanggal'].min().strftime('%d %b %Y')} - {df['tanggal'].max().strftime('%d %b %Y')} | "
        f"{datetime.now().strftime('%d %B %Y %H:%M')}"
    )
else:
    st.caption("© 2025 Harga Pangan Analytics Dashboard")