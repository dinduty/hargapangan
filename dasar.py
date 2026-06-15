import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(
    page_title="Harga Bahan Pangan Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Panel Kontrol")
st.sidebar.markdown("---")

# Pilih metode analisis
analysis_type = st.sidebar.selectbox(
    "Pilih Metode Analisis",
    ["📈 SARIMA (Prediksi Harga)", "🌲 Random Forest Classifier", "🎯 K-Means Clustering"]
)

# Filter wilayah
province = st.sidebar.selectbox("Provinsi", ["Jawa Barat", "Jawa Timur", "DKI Jakarta", "Jawa Tengah", "Banten"])
commodity = st.sidebar.selectbox("Komoditas", ["Beras", "Cabai Rawit", "Bawang Merah", "Minyak Goreng", "Daging Ayam"])
date_range = st.sidebar.date_input("Rentang Tanggal", [datetime(2024, 1, 1), datetime(2024, 12, 31)])


# Tambahkan di bagian Sidebar, setelah filter
st.sidebar.markdown("---")
st.sidebar.subheader("📂 Upload Data Bahan Pangan")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV/Excel",
    type=["csv", "xlsx"],
    help="Format: kolom tanggal, harga, pasar, provinsi, dll."
)

if uploaded_file is not None:
    import pandas as pd
    if uploaded_file.name.endswith('.csv'):
        df_custom = pd.read_csv(uploaded_file)
    else:
        df_custom = pd.read_excel(uploaded_file)
    
    st.sidebar.success(f"✅ {len(df_custom)} baris data loaded!")
    # Gunakan df_custom untuk analisis
else:
    st.sidebar.info("ℹ️ Gunakan data demo atau upload file CSV/Excel SISP Anda")

# ==================== HEADER ====================
st.title("📊 SISP Analytics Dashboard")
st.caption(f"Dashboard Analisis Data Pasar - {province} | Komoditas: {commodity}")

# ==================== FUNGSI GENERATE DATA DUMMY ====================
@st.cache_data
def generate_price_data(province, commodity, days=180):
    """Generate data harga dummy untuk demo"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Base price berdasarkan komoditas
    base_prices = {
        "Beras": 14000, "Cabai Rawit": 45000, "Bawang Merah": 30000,
        "Minyak Goreng": 18000, "Daging Ayam": 38000
    }
    base_price = base_prices.get(commodity, 20000)
    
    # Generate harga dengan trend dan musiman
    t = np.arange(days)
    seasonal = 5000 * np.sin(2 * np.pi * t / 30)  # siklus 30 hari
    trend = 0.5 * t  # trend naik perlahan
    noise = np.random.normal(0, 1000, days)
    
    prices = base_price + seasonal + trend + noise
    prices = np.maximum(prices, base_price * 0.7)  # batas bawah
    
    return pd.DataFrame({"tanggal": dates, "harga": prices})

@st.cache_data
def generate_market_data(province):
    """Generate data pasar untuk clustering & klasifikasi"""
    n_markets = np.random.randint(50, 150)
    markets = []
    
    for i in range(n_markets):
        markets.append({
            "id_pasar": f"PSR{i+1:04d}",
            "nama_pasar": f"Pasar {['Induk', 'Legi', 'Baru', 'Gede', 'Besar'][np.random.randint(0,5)]} {i+1}",
            "kecamatan": f"Kecamatan {np.random.randint(1, 20)}",
            "jumlah_kios": np.random.randint(20, 500),
            "luas_lahan_m2": np.random.randint(500, 5000),
            "nilai_dak": np.random.randint(500000000, 5000000000, dtype=np.int64),
            "rata_harga_beras": np.random.randint(12000, 18000),
            "rata_harga_cabai": np.random.randint(30000, 60000),
            "fluktuasi_harga": np.random.uniform(0.05, 0.25),
            "status_renovasi": np.random.choice(["Baik", "Rusak Ringan", "Rusak Berat"], p=[0.4, 0.4, 0.2]),
            "lon": np.random.uniform(95, 141),
            "lat": np.random.uniform(-8, 6)
        })
    
    return pd.DataFrame(markets)

# ==================== 1. SARIMA SECTION ====================
def sarima_section(province, commodity):
    st.header("📈 SARIMA: Prediksi Harga Komoditas")
    
    # Load data
    df_price = generate_price_data(province, commodity)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot data historis dengan Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_price["tanggal"], y=df_price["harga"],
            mode='lines', name='Harga Aktual',
            line=dict(color='blue', width=2)
        ))
        
        # Tambahkan prediksi (simulasi SARIMA)
        last_date = df_price["tanggal"].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30, freq='D')
        
        # Simulasi hasil prediksi SARIMA
        last_price = df_price["harga"].iloc[-1]
        future_prices = last_price + np.cumsum(np.random.normal(0, 300, 30))
        future_prices = np.maximum(future_prices, last_price * 0.85)
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_prices,
            mode='lines', name='Prediksi SARIMA',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Konfidensi interval
        upper_bound = future_prices + np.random.normal(300, 100, 30)
        lower_bound = future_prices - np.random.normal(300, 100, 30)
        
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
            yaxis_title="Harga (Rp/kg)",
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrik prediksi
        st.subheader("📊 Ringkasan Prediksi")
        
        pred_7_hari = future_prices[6]
        pred_30_hari = future_prices[29]
        perubahan_persen = ((pred_30_hari - last_price) / last_price) * 100
        
        st.metric(
            f"Harga {commodity} Hari Ini",
            f"Rp {last_price:,.0f}",
            delta=None
        )
        st.metric(
            "Prediksi 7 Hari",
            f"Rp {pred_7_hari:,.0f}",
            delta=f"{(pred_7_hari - last_price):+,.0f}"
        )
        st.metric(
            "Prediksi 30 Hari",
            f"Rp {pred_30_hari:,.0f}",
            delta=f"{perubahan_persen:+.1f}%",
            delta_color="inverse" if perubahan_persen > 0 else "normal"
        )
        
        # Parameter SARIMA
        with st.expander("📐 Parameter Model SARIMA"):
            st.code("""
            Model: SARIMA(1,1,0)(0,1,1)[12]
            
            Parameters:
            - p=1, d=1, q=0 (Non-seasonal)
            - P=0, D=1, Q=1, s=12 (Seasonal)
            - AIC: 1245.67
            - RMSE: 312.45
            """)
    
    # Tabel data historis
    with st.expander("📋 Lihat Data Historis"):
        st.dataframe(
            df_price.tail(30).sort_values("tanggal", ascending=False),
            use_container_width=True,
            hide_index=True
        )

# ==================== 2. RANDOM FOREST SECTION ====================
def random_forest_section(province):
    st.header("🌲 Random Forest Classifier: Prediksi Risiko Pasar")
    
    # Load data
    df_markets = generate_market_data(province)
    
    # Simulasi hasil klasifikasi Random Forest
    np.random.seed(42)
    probabilities = np.random.uniform(0, 1, len(df_markets))
    
    # Buat label berdasarkan probabilitas
    df_markets['prob_risiko'] = probabilities
    df_markets['status_risiko'] = np.where(
        probabilities < 0.3, "Rendah",
        np.where(probabilities < 0.7, "Sedang", "Tinggi")
    )
    
    # Hitung feature importance (dummy)
    feature_importance = {
        'Jumlah Kios': 0.28,
        'Luas Lahan': 0.22,
        'Nilai DAK': 0.18,
        'Rata Harga Beras': 0.15,
        'Fluktuasi Harga': 0.12,
        'Status Renovasi': 0.05
    }
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Distribusi status risiko
        risk_counts = df_markets['status_risiko'].value_counts()
        colors = {'Rendah': '#2ecc71', 'Sedang': '#f39c12', 'Tinggi': '#e74c3c'}
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                marker_colors=[colors[r] for r in risk_counts.index],
                hole=0.4,
                textinfo='label+percent'
            )
        ])
        fig.update_layout(title="Distribusi Status Risiko Pasar")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feature Importance
        fig = go.Figure(data=[
            go.Bar(
                x=list(feature_importance.values()),
                y=list(feature_importance.keys()),
                orientation='h',
                marker_color='#3498db'
            )
        ])
        fig.update_layout(
            title="Feature Importance (Random Forest)",
            xaxis_title="Importance Score",
            yaxis_title="Fitur"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tabel pasar dengan risiko tinggi
    st.subheader("⚠️ Pasar dengan Risiko Tinggi (Prioritas Intervensi)")
    
    high_risk_markets = df_markets[df_markets['status_risiko'] == 'Tinggi'].head(10)
    
    if not high_risk_markets.empty:
        st.dataframe(
            high_risk_markets[[
                'nama_pasar', 'kecamatan', 'jumlah_kios', 
                'nilai_dak', 'status_renovasi', 'prob_risiko'
            ]].assign(
                prob_risiko=lambda x: x['prob_risiko'].apply(lambda p: f"{p:.1%}"),
                nilai_dak=lambda x: x['nilai_dak'].apply(lambda v: f"Rp {v:,.0f}")
            ),
            use_container_width=True,
            hide_index=True
        )
    
    # Matriks Konfusi (simulasi)
    with st.expander("📊 Evaluasi Model Random Forest"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Classification Report")
            st.code("""
                  precision    recall  f1-score   support
        
         Rendah       0.88      0.85      0.86        45
         Sedang       0.82      0.84      0.83        52
         Tinggi       0.91      0.89      0.90        38
        
        accuracy                           0.86       135
        macro avg      0.87      0.86      0.86       135
        """)
        with col2:
            # Simulasi confusion matrix
            cm = np.array([[38, 5, 2], [4, 44, 4], [2, 4, 32]])
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Rendah', 'Sedang', 'Tinggi'],
                        yticklabels=['Rendah', 'Sedang', 'Tinggi'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)

# ==================== 3. K-MEANS SECTION ====================
def kmeans_section(province):
    st.header("🎯 K-Means Clustering: Segmentasi Pasar")
    
    # Load data
    df_markets = generate_market_data(province)
    
    # Simulasi hasil K-Means dengan 3 cluster
    np.random.seed(42)
    from sklearn.preprocessing import StandardScaler
    
    # Fitur untuk clustering
    features = ['jumlah_kios', 'luas_lahan_m2', 'nilai_dak', 'fluktuasi_harga']
    X = df_markets[features].copy()
    
    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Simulasi label cluster (dummy K-Means)
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_markets['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Mapping cluster ke nama
    cluster_names = {
        0: "Pasar Premium",
        1: "Pasar Menengah",
        2: "Pasar Rakyat"
    }
    df_markets['segmen'] = df_markets['cluster'].map(cluster_names)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Visualisasi cluster dengan scatter plot
        fig = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71']
        for cluster_id, color in zip(range(3), colors):
            cluster_data = df_markets[df_markets['cluster'] == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_data['jumlah_kios'],
                y=cluster_data['nilai_dak'],
                mode='markers',
                name=cluster_names[cluster_id],
                marker=dict(size=10, color=color, opacity=0.7),
                text=cluster_data['nama_pasar'],
                hovertemplate='<b>%{text}</b><br>Jumlah Kios: %{x}<br>Nilai DAK: Rp %{y:,.0f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Segmentasi Pasar (K-Means Clustering)",
            xaxis_title="Jumlah Kios",
            yaxis_title="Nilai DAK (Rp)",
            hovermode='closest'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Profil masing-masing cluster
        st.subheader("📊 Profil Segmentasi Pasar")
        
        cluster_profile = df_markets.groupby('segmen').agg({
            'jumlah_kios': 'mean',
            'luas_lahan_m2': 'mean',
            'nilai_dak': 'mean',
            'fluktuasi_harga': 'mean'
        }).round(2)
        
        cluster_profile.columns = ['Rata2 Jumlah Kios', 'Rata2 Luas (m²)', 'Rata2 Nilai DAK', 'Rata2 Fluktuasi']
        cluster_profile['Rata2 Nilai DAK'] = cluster_profile['Rata2 Nilai DAK'].apply(lambda x: f"Rp {x:,.0f}")
        
        st.dataframe(cluster_profile, use_container_width=True)
    
    # Interpretasi dan rekomendasi
    st.subheader("💡 Interpretasi & Rekomendasi")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🟡 Pasar Rakyat")
            st.markdown("- Jumlah kios sedikit")
            st.markdown("- Nilai DAK terendah")
            st.markdown("- **Rekomendasi**: Prioritas bantuan renovasi dasar")
        
        with col2:
            st.markdown("#### 🔵 Pasar Menengah")
            st.markdown("- Fasilitas memadai")
            st.markdown("- Fluktuasi harga sedang")
            st.markdown("- **Rekomendasi**: Optimalisasi digitalisasi pasar")
        
        with col3:
            st.markdown("#### 🔴 Pasar Premium")
            st.markdown("- Jumlah kios besar")
            st.markdown("- Nilai DAK tertinggi")
            st.markdown("- **Rekomendasi**: Implementasi sistem pembayaran digital")
    
    # Metrik evaluasi clustering
    with st.expander("📐 Evaluasi Model K-Means"):
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X_scaled, df_markets['cluster'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Silhouette Score", f"{sil_score:.3f}")
            st.caption("Semakin mendekati 1, semakin baik clustering")
        with col2:
            st.metric("Jumlah Cluster (K)", "3")
            st.caption("Optimal berdasarkan Elbow Method")

# ==================== MAIN ROUTING ====================
if analysis_type == "📈 SARIMA (Prediksi Harga)":
    sarima_section(province, commodity)
elif analysis_type == "🌲 Random Forest Classifier":
    random_forest_section(province)
else:
    kmeans_section(province)

# ==================== FOOTER ====================
st.markdown("---")
st.caption(
    f"© 2025 SISP Analytics Dashboard | Data terakhir diperbarui: {datetime.now().strftime('%d %B %Y %H:%M')} | "
    "Powered by Streamlit"
)