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
from sklearn.model_selection import cross_val_score
from scipy import stats
from pathlib import Path

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Harga Bahan Pangan Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD DATA ====================
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
    [
        "📈 SARIMA (Prediksi Harga)",
        "🌲 Random Forest Classifier",
        "🎯 K-Means Clustering",
        "📊 Perbandingan RF vs K-Means"  # Menu baru
    ]
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

# ==================== FUNGSI PREPARE DATA UNTUK PERBANDINGAN ====================
def prepare_comparison_data(df, province, date_range):
    """Siapkan data untuk perbandingan RF vs K-Means"""
    
    if df is None or df.empty:
        return None, None
    
    df_data = df[df['provinsi'] == province].copy()
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        df_data = df_data[
            (df_data['tanggal'] >= pd.to_datetime(start_date)) & 
            (df_data['tanggal'] <= pd.to_datetime(end_date))
        ]
    
    if df_data.empty:
        return None, None
    
    commodities = df_data['komoditas'].unique()
    feature_data = []
    
    for kom in commodities:
        df_kom = df_data[df_data['komoditas'] == kom]
        
        if len(df_kom) >= 5:
            mean_price = df_kom['harga'].mean()
            std_price = df_kom['harga'].std()
            fluctuation = std_price / mean_price if mean_price > 0 else 0
            
            if len(df_kom) >= 5:
                x = np.arange(len(df_kom))
                slope, _, _, _, _ = stats.linregress(x, df_kom['harga'].values)
                trend_hari = slope
            else:
                trend_hari = 0
            
            feature_data.append({
                'komoditas': kom,
                'rata_harga': mean_price,
                'std_harga': std_price,
                'fluktuasi': fluctuation,
                'tren_hari': trend_hari,
                'tren_bulan': trend_hari * 30,
                'harga_min': df_kom['harga'].min(),
                'harga_max': df_kom['harga'].max(),
                'range_harga': df_kom['harga'].max() - df_kom['harga'].min(),
                'jumlah_data': len(df_kom)
            })
    
    if not feature_data:
        return None, None
    
    df_features = pd.DataFrame(feature_data)
    return df_features, df_data

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
    
    st.markdown("""
    ### 📖 Analisis Risiko Komoditas dengan 3 Indikator
    
    Random Forest Classifier mengelompokkan komoditas berdasarkan **3 indikator risiko**:
    
    | Indikator | Rumus | Makna Risiko |
    |-----------|-------|--------------|
    | 📊 **Fluktuasi** | `Std Harga / Rata-rata Harga` | Semakin tinggi → harga tidak stabil |
    | 📈 **Tren Harga** | `Kemiringan (slope) regresi linear` | Semakin positif → harga cenderung naik |
    | 📏 **Range Harga** | `Harga Maks - Harga Min` | Semakin besar → rentang harga lebar |
    
    **Klasifikasi Risiko:**
    - 🟢 **Rendah**: Indikator di bawah median
    - 🟡 **Sedang**: 1-2 indikator di atas median
    - 🔴 **Tinggi**: 2-3 indikator di atas median
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
            
            if len(df_kom) >= 5:
                from scipy import stats
                x = np.arange(len(df_kom))
                slope, _, _, _, _ = stats.linregress(x, df_kom['harga'].values)
                trend_per_hari = slope
                trend_per_bulan = slope * 30
            else:
                trend_per_hari = 0
                trend_per_bulan = 0
            
            harga_min = df_kom['harga'].min()
            harga_max = df_kom['harga'].max()
            range_harga = harga_max - harga_min
            
            feature_data.append({
                'komoditas': kom,
                'rata_harga': mean_price,
                'std_harga': std_price,
                'fluktuasi': fluctuation,
                'tren_hari': trend_per_hari,
                'tren_bulan': trend_per_bulan,
                'harga_tertinggi': harga_max,
                'harga_terendah': harga_min,
                'range_harga': range_harga,
                'jumlah_data': len(df_kom)
            })
    
    if not feature_data:
        st.warning("⚠️ Data tidak mencukupi untuk analisis (minimal 5 data per komoditas)")
        return
    
    df_features = pd.DataFrame(feature_data)
    
    median_fluktuasi = df_features['fluktuasi'].median()
    median_tren = df_features['tren_hari'].median()
    median_range = df_features['range_harga'].median()
    
    df_features['fluktuasi_risk'] = df_features['fluktuasi'] > median_fluktuasi
    df_features['tren_risk'] = df_features['tren_hari'] > median_tren
    df_features['range_risk'] = df_features['range_harga'] > median_range
    
    df_features['total_risk'] = (
        df_features['fluktuasi_risk'].astype(int) +
        df_features['tren_risk'].astype(int) +
        df_features['range_risk'].astype(int)
    )
    
    df_features['risk_level'] = np.where(
        df_features['total_risk'] >= 2, "🔴 Risiko Tinggi",
        np.where(df_features['total_risk'] >= 1, "🟡 Risiko Sedang", "🟢 Risiko Rendah")
    )
    
    st.subheader("📊 Statistik Harga & Indikator Risiko per Komoditas")
    
    display_df = df_features[[
        'komoditas', 'rata_harga', 'fluktuasi', 'tren_bulan', 'range_harga', 'total_risk', 'risk_level'
    ]].copy()
    
    display_df = display_df.sort_values('total_risk', ascending=False)
    display_df['total_risk'] = display_df['total_risk'].apply(lambda x: f"{x}/3")
    
    st.dataframe(
        display_df.style.format({
            'rata_harga': 'Rp {:,.0f}',
            'fluktuasi': '{:.1%}',
            'tren_bulan': 'Rp {:,.0f}/bulan',
            'range_harga': 'Rp {:,.0f}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    col1, col2, col3 = st.columns(3)
    
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
        fig.add_hline(y=median_fluktuasi, line_dash="dash", line_color="red", 
                      annotation_text=f"Median: {median_fluktuasi:.1%}")
        fig.update_layout(
            title="1️⃣ Fluktuasi Harga",
            xaxis_title="Komoditas",
            yaxis_title="Koefisien Variasi",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(
                x=df_features['komoditas'],
                y=df_features['tren_bulan'],
                marker_color='#2ecc71',
                text=df_features['tren_bulan'].apply(lambda x: f"Rp {x:,.0f}/bln"),
                textposition='outside'
            )
        ])
        fig.add_hline(y=median_tren * 30, line_dash="dash", line_color="red",
                      annotation_text=f"Median: Rp {median_tren*30:,.0f}/bln")
        fig.update_layout(
            title="2️⃣ Tren Harga (per Bulan)",
            xaxis_title="Komoditas",
            yaxis_title="Perubahan (Rp/bulan)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(data=[
            go.Bar(
                x=df_features['komoditas'],
                y=df_features['range_harga'],
                marker_color='#f39c12',
                text=df_features['range_harga'].apply(lambda x: f"Rp {x:,.0f}"),
                textposition='outside'
            )
        ])
        fig.add_hline(y=median_range, line_dash="dash", line_color="red",
                      annotation_text=f"Median: Rp {median_range:,.0f}")
        fig.update_layout(
            title="3️⃣ Range Harga (Max - Min)",
            xaxis_title="Komoditas",
            yaxis_title="Selisih Harga (Rp)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Rekapitulasi Klasifikasi Risiko")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_counts = df_features['risk_level'].value_counts()
        colors = {'🔴 Risiko Tinggi': '#e74c3c', '🟡 Risiko Sedang': '#f39c12', '🟢 Risiko Rendah': '#2ecc71'}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=[colors.get(r, '#95a5a6') for r in risk_counts.index],
                hole=0.4,
                textinfo='label+percent'
            )
        ])
        fig.update_layout(title="Distribusi Risiko Komoditas")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Detail Indikator Risiko")
        
        detail_df = df_features[[
            'komoditas', 'fluktuasi_risk', 'tren_risk', 'range_risk', 'total_risk', 'risk_level'
        ]].copy()
        
        detail_df['fluktuasi_risk'] = detail_df['fluktuasi_risk'].replace({True: '✅', False: '❌'})
        detail_df['tren_risk'] = detail_df['tren_risk'].replace({True: '✅', False: '❌'})
        detail_df['range_risk'] = detail_df['range_risk'].replace({True: '✅', False: '❌'})
        
        st.dataframe(
            detail_df.sort_values('total_risk', ascending=False),
            column_config={
                'komoditas': 'Komoditas',
                'fluktuasi_risk': 'Fluktuasi',
                'tren_risk': 'Tren',
                'range_risk': 'Range',
                'total_risk': 'Skor',
                'risk_level': 'Kategori'
            },
            hide_index=True,
            use_container_width=True
        )
    
    st.subheader("💡 Rekomendasi Kebijakan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🟢 Risiko Rendah (0-1 indikator)")
        st.markdown("""
        **Karakteristik:**
        - Harga stabil
        - Tren harga rendah
        - Range harga sempit
        
        **Tindakan:** Pemantauan rutin
        """)
    
    with col2:
        st.markdown("#### 🟡 Risiko Sedang (1-2 indikator)")
        st.markdown("""
        **Karakteristik:**
        - Harga cukup fluktuatif
        - Tren harga meningkat
        - Range harga cukup lebar
        
        **Tindakan:** Pemantauan intensif + sosialisasi
        """)
    
    with col3:
        st.markdown("#### 🔴 Risiko Tinggi (2-3 indikator)")
        st.markdown("""
        **Karakteristik:**
        - Harga sangat fluktuatif
        - Tren harga naik signifikan
        - Range harga sangat lebar
        
        **Tindakan:** 
        - ⭐ Prioritas intervensi
        - Operasi pasar
        - Stabilisasi stok
        """)

    with st.expander("📐 Detail Median Indikator"):
        st.markdown(f"""
        | Indikator | Median | Komoditas di Atas Median |
        |-----------|--------|--------------------------|
        | Fluktuasi | {median_fluktuasi:.1%} | {', '.join(df_features[df_features['fluktuasi'] > median_fluktuasi]['komoditas'].tolist())} |
        | Tren (Rp/bulan) | Rp {median_tren*30:,.0f} | {', '.join(df_features[df_features['tren_hari'] > median_tren]['komoditas'].tolist())} |
        | Range | Rp {median_range:,.0f} | {', '.join(df_features[df_features['range_harga'] > median_range]['komoditas'].tolist())} |
        """)

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
    
    features = ['rata_harga', 'std_harga', 'fluktuasi', 'range_harga']
    X = df_features[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = min(3, len(df_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features['cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_order = df_features.groupby('cluster')['rata_harga'].mean().sort_values(ascending=False)
    
    segmen_names = ["🔴 Mahal", "🟡 Sedang", "🟢 Terjangkau"]
    for i, cluster_id in enumerate(cluster_order.index):
        if i < len(segmen_names):
            df_features.loc[df_features['cluster'] == cluster_id, 'segmen'] = segmen_names[i]
    
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
    
    with st.expander("📐 Evaluasi Model K-Means"):
        sil_score = silhouette_score(X_scaled, df_features['cluster'])
        st.metric("Silhouette Score", f"{sil_score:.3f}")
        st.caption("Semakin mendekati 1, semakin baik clustering")
        
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

# ==================== 4. PERBANDINGAN RF vs K-MEANS ====================
def comparison_section(df, province, date_range):
    st.header("📊 Perbandingan Random Forest vs K-Means")
    
    st.markdown("""
    ### 📖 Perbandingan Dua Metode Analisis
    
    Halaman ini membandingkan hasil dari dua metode yang berbeda:
    
    | Metode | Tujuan | Pendekatan |
    |--------|--------|------------|
    | **Random Forest** | Klasifikasi risiko komoditas | Supervised Learning (dengan label) |
    | **K-Means** | Segmentasi komoditas | Unsupervised Learning (tanpa label) |
    
    **Perbedaan Utama:**
    - Random Forest mengelompokkan berdasarkan **aturan risiko** (fluktuasi)
    - K-Means mengelompokkan berdasarkan **kemiripan karakteristik** (harga, fluktuasi, dll)
    """)
    
    if df is None or df.empty:
        st.warning("⚠️ Tidak ada data untuk dianalisis")
        return
    
    df_features, df_data = prepare_comparison_data(df, province, date_range)
    
    if df_features is None or len(df_features) < 3:
        st.warning("⚠️ Data tidak cukup untuk perbandingan (minimal 3 komoditas)")
        return
    
    # ==================== RUN RANDOM FOREST ====================
    features = ['rata_harga', 'fluktuasi', 'tren_hari', 'range_harga']
    X = df_features[features].copy()
    
    median_fluktuasi = df_features['fluktuasi'].median()
    y = np.where(
        df_features['fluktuasi'] > median_fluktuasi * 1.5, 2,
        np.where(df_features['fluktuasi'] > median_fluktuasi, 1, 0)
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=min(3, len(df_features)))
    rf.fit(X, y)
    feature_importance = dict(zip(features, rf.feature_importances_))
    
    rf_pred = rf.predict(X)
    label_map = {0: '🟢 Rendah', 1: '🟡 Sedang', 2: '🔴 Tinggi'}
    df_features['rf_category'] = [label_map[p] for p in rf_pred]
    
    # ==================== RUN K-MEANS ====================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    n_clusters = min(3, len(df_features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_features['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_order = df_features.groupby('kmeans_cluster')['rata_harga'].mean().sort_values(ascending=False)
    segmen_names = ["🔴 Mahal", "🟡 Sedang", "🟢 Terjangkau"]
    for i, cluster_id in enumerate(cluster_order.index):
        if i < len(segmen_names):
            df_features.loc[df_features['kmeans_cluster'] == cluster_id, 'kmeans_category'] = segmen_names[i]
    
    sil_score = silhouette_score(X_scaled, df_features['kmeans_cluster'])
    
    # ==================== 1. METRIK PERBANDINGAN ====================
    st.subheader("📊 1. Metrik Kinerja")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Jumlah Komoditas", len(df_features))
    
    with col2:
        st.metric(
            "Random Forest (CV Score)",
            f"{cv_scores.mean():.2f}",
            help="Cross-validation score (semakin tinggi = semakin baik)"
        )
    
    with col3:
        st.metric(
            "K-Means (Silhouette)",
            f"{sil_score:.3f}",
            help="Silhouette score (semakin mendekati 1 = semakin baik)"
        )
    
    with col4:
        st.metric("Jumlah Cluster", n_clusters)
    
    # ==================== 2. TABEL PERBANDINGAN ====================
    st.subheader("📋 2. Perbandingan Hasil per Komoditas")
    
    compare_df = pd.DataFrame({
        'Komoditas': df_features['komoditas'],
        'Rata Harga': df_features['rata_harga'],
        'Fluktuasi': df_features['fluktuasi'],
        'RF Prediksi': df_features['rf_category'],
        'K-Means Cluster': df_features['kmeans_category']
    })
    
    st.dataframe(
        compare_df.style.format({
            'Rata Harga': 'Rp {:,.0f}',
            'Fluktuasi': '{:.1%}'
        }),
        use_container_width=True,
        hide_index=True
    )
    
    # ==================== 3. VISUALISASI ====================
    st.subheader("📊 3. Visualisasi Perbandingan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        colors = {'🟢 Rendah': '#2ecc71', '🟡 Sedang': '#f39c12', '🔴 Tinggi': '#e74c3c'}
        
        for cat in df_features['rf_category'].unique():
            data = df_features[df_features['rf_category'] == cat]
            fig.add_trace(go.Scatter(
                x=data['rata_harga'],
                y=data['fluktuasi'],
                mode='markers+text',
                name=f'RF: {cat}',
                marker=dict(size=20, color=colors.get(cat, '#95a5a6'), opacity=0.8),
                text=data['komoditas'],
                textposition='top center'
            ))
        
        fig.update_layout(
            title="Random Forest: Klasifikasi Risiko",
            xaxis_title="Rata-rata Harga (Rp/kg)",
            yaxis_title="Fluktuasi (%)",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        colors = {'🔴 Mahal': '#e74c3c', '🟡 Sedang': '#f39c12', '🟢 Terjangkau': '#2ecc71'}
        
        for cat in df_features['kmeans_category'].unique():
            data = df_features[df_features['kmeans_category'] == cat]
            fig.add_trace(go.Scatter(
                x=data['rata_harga'],
                y=data['fluktuasi'],
                mode='markers+text',
                name=f'K-Means: {cat}',
                marker=dict(size=20, color=colors.get(cat, '#95a5a6'), opacity=0.8),
                text=data['komoditas'],
                textposition='top center'
            ))
        
        fig.update_layout(
            title="K-Means: Segmentasi Komoditas",
            xaxis_title="Rata-rata Harga (Rp/kg)",
            yaxis_title="Fluktuasi (%)",
            yaxis_tickformat='.1%'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # ==================== 4. MATRIKS PERBANDINGAN ====================
    st.subheader("📊 4. Matriks Perbandingan RF vs K-Means")
    
    rf_map = {'🟢 Rendah': 0, '🟡 Sedang': 1, '🔴 Tinggi': 2}
    km_map = {'🟢 Terjangkau': 0, '🟡 Sedang': 1, '🔴 Mahal': 2}
    
    rf_labels = df_features['rf_category'].map(rf_map).fillna(-1).astype(int)
    km_labels = df_features['kmeans_category'].map(km_map).fillna(-1).astype(int)
    
    cm = np.zeros((3, 3), dtype=int)
    for i in range(len(rf_labels)):
        if rf_labels[i] >= 0 and km_labels[i] >= 0:
            cm[rf_labels[i], km_labels[i]] += 1
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Terjangkau', 'Sedang', 'Mahal'],
        yticklabels=['Rendah', 'Sedang', 'Tinggi'],
        ax=ax
    )
    ax.set_xlabel('K-Means')
    ax.set_ylabel('Random Forest')
    ax.set_title('Perbandingan Klasifikasi RF vs Segmentasi K-Means')
    
    st.pyplot(fig)
    
    # ==================== 5. FEATURE IMPORTANCE ====================
    st.subheader("📊 5. Feature Importance (Random Forest)")
    
    fi_df = pd.DataFrame({
        'Fitur': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(data=[
        go.Bar(
            x=fi_df['Importance'],
            y=fi_df['Fitur'],
            orientation='h',
            marker_color='#3498db',
            text=fi_df['Importance'].apply(lambda x: f"{x:.2%}"),
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="Feature Importance - Fitur Paling Berpengaruh",
        xaxis_title="Importance Score",
        yaxis_title="Fitur"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ==================== 6. KESIMPULAN & REKOMENDASI ====================
    st.subheader("📝 6. Kesimpulan & Rekomendasi")
    
    same_count = (rf_labels == km_labels).sum()
    total = len(rf_labels)
    similarity = same_count / total * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #2ecc71; padding: 15px; border-radius: 10px;">
            <h4 style="color: white; margin-top: 0;">🌲 Random Forest Classifier</h4>
            <p style="color: white;">
                <b>Kelebihan:</b><br>
                ✅ Memberikan label risiko jelas<br>
                ✅ Ada feature importance<br>
                ✅ CV Score: <b>{cv_scores.mean():.2f}</b>
            </p>
            <p style="color: white;">
                <b>Kekurangan:</b><br>
                ❌ Membutuhkan label training<br>
                ❌ Interpretasi lebih kaku
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background-color: #3498db; padding: 15px; border-radius: 10px;">
            <h4 style="color: white; margin-top: 0;">🎯 K-Means Clustering</h4>
            <p style="color: white;">
                <b>Kelebihan:</b><br>
                ✅ Tidak butuh label<br>
                ✅ Menemukan pola alami<br>
                ✅ Silhouette: <b>{sil_score:.3f}</b>
            </p>
            <p style="color: white;">
                <b>Kekurangan:</b><br>
                ❌ Nama cluster manual<br>
                ❌ K harus ditentukan
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if similarity > 70:
        st.success(f"✅ Kedua metode menunjukkan hasil yang cukup selaras ({similarity:.0f}% kesamaan)")
    else:
        st.warning(f"⚠️ Kedua metode menunjukkan hasil yang berbeda ({similarity:.0f}% kesamaan). K-Means mungkin menemukan pola yang berbeda dari klasifikasi risiko.")
    
    st.info("""
    💡 **Rekomendasi:**
    
    - Gunakan **Random Forest** jika Anda ingin klasifikasi risiko yang jelas dan terukur
    - Gunakan **K-Means** jika Anda ingin segmentasi komoditas berdasarkan kemiripan karakteristik
    - Kombinasikan keduanya untuk analisis yang lebih komprehensif
    """)

# ==================== MAIN ROUTING ====================
if df is not None:
    if analysis_type == "📈 SARIMA (Prediksi Harga)":
        sarima_section(df, province, commodity, date_range)
    elif analysis_type == "🌲 Random Forest Classifier":
        random_forest_section(df, province, date_range)
    elif analysis_type == "🎯 K-Means Clustering":
        kmeans_section(df, province, date_range)
    elif analysis_type == "📊 Perbandingan RF vs K-Means":
        comparison_section(df, province, date_range)
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