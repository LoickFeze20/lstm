import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Configuration de la page
st.set_page_config(
    page_title="Apple LSTM Predictor",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé (identique à l'autre app)
st.markdown("""
<style>
    /* Style général */
    .main-header {
        background: linear-gradient(90deg, #1E3C72 0%, #2A5298 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 600;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
    }
    
    /* Cartes métriques */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
        border: 1px solid #eaeaea;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .metric-card h3 {
        color: #666;
        font-size: 0.9rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 600;
        color: #1E3C72;
        margin: 0.5rem 0;
    }
    .metric-card .delta {
        font-size: 0.9rem;
        color: #28a745;
    }
    .metric-card .delta.negative {
        color: #dc3545;
    }
    
    /* Boutons de navigation */
    .nav-button {
        background: white;
        border: 2px solid #1E3C72;
        color: #1E3C72;
        padding: 0.75rem;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s;
        margin: 0.25rem 0;
        font-weight: 500;
    }
    .nav-button:hover {
        background: #1E3C72;
        color: white;
    }
    .nav-button.active {
        background: #1E3C72;
        color: white;
    }
    
    /* Conteneurs */
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #1E3C72;
        margin: 1rem 0;
    }
    
    /* Tableau */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #eaeaea;
    }
</style>
""", unsafe_allow_html=True)

# Session state pour la navigation
if 'page' not in st.session_state:
    st.session_state.page = 'Accueil'

# Sidebar navigation
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='color: #1E3C72;'>🍎 Apple LSTM</h2>
        <p style='color: #666;'>Deep Learning Predictor</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Boutons de navigation
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home", help="Accueil", use_container_width=True):
            st.session_state.page = 'Accueil'
    with col2:
        if st.button("📊 Pred", help="Prédictions", use_container_width=True):
            st.session_state.page = 'Prédictions'
    with col3:
        if st.button("ℹ️ Infos", help="Informations", use_container_width=True):
            st.session_state.page = 'Infos'
    
    st.markdown("---")
    
    # 🔗 LIEN VERS L'AUTRE APPLICATION (NeuralProphet)
    with st.expander("🔗 NeuralProphet"):
        st.markdown("""
        <a href="https://neuralprohet-7.streamlit.app" target="_blank">
            <button style="
                width: 100%;
                padding: 0.75rem;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                margin: 0.5rem 0;
            ">
                🚀 App NeuralProphet
            </button>
        </a>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chargement du modèle LSTM
    @st.cache_resource
    def load_lstm_model():
        try:
            if os.path.exists("apple_lstm.h5"):
                model = load_model("apple_lstm.h5")
                return model
            else:
                return None
        except Exception as e:
            st.error(f"Erreur chargement: {e}")
            return None
    
    model = load_lstm_model()
    
    if model:
        st.success("✅ Modèle LSTM chargé")
        with st.expander("📦 Architecture"):
            model.summary(print_fn=lambda x: st.text(x))
    else:
        st.error("❌ Modèle 'apple_lstm.h5' non trouvé")
    
    st.markdown("---")
    
    # Paramètres
    st.subheader("⚙️ Paramètres")
    window_size = st.number_input("Fenêtre (jours)", 10, 100, 60, 
                                 help="Nombre de jours d'historique pour prédire")
    jours = st.slider("Jours à prédire", 1, 90, 30)

# Fonctions de preprocessing pour LSTM
def create_sequences(data, window_size):
    """Crée des séquences pour LSTM"""
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def prepare_data_for_lstm(df, window_size):
    """Prépare les données pour LSTM"""
    from sklearn.preprocessing import MinMaxScaler
    
    # Prendre les prix de clôture
    prices = df['Close'].values.reshape(-1, 1)
    
    # Normalisation
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(prices)
    
    return scaled_prices, scaler

def predict_future(model, last_sequence, scaler, days):
    """Prédit les jours futurs"""
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(days):
        # Reshape pour LSTM (batch_size, timesteps, features)
        current_seq_reshaped = current_seq.reshape(1, current_seq.shape[0], 1)
        
        # Prédire le prochain jour
        next_pred = model.predict(current_seq_reshaped, verbose=0)[0, 0]
        predictions.append(next_pred)
        
        # Mettre à jour la séquence
        current_seq = np.append(current_seq[1:], next_pred)
    
    # Dénormaliser
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()

# Chargement des données
@st.cache_data
def load_data():
    apple = yf.Ticker("AAPL")
    hist = apple.history(period="2y")
    hist.reset_index(inplace=True)
    return hist

data = load_data()

# PAGE D'ACCUEIL
if st.session_state.page == 'Accueil':
    st.markdown("""
    <div class='main-header'>
        <h1>🍎 Apple LSTM Predictor</h1>
        <p>Prédictions avec réseau de neurones LSTM</p>
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    week_ago = data['Close'].iloc[-6] if len(data) > 6 else current_price
    month_ago = data['Close'].iloc[-21] if len(data) > 21 else current_price
    
    with col1:
        delta_day = ((current_price - prev_price) / prev_price * 100)
        delta_class = "delta negative" if delta_day < 0 else "delta"
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Prix actuel</h3>
            <div class='value'>${current_price:.2f}</div>
            <div class='{delta_class}'>Jour: {delta_day:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_week = ((current_price - week_ago) / week_ago * 100)
        delta_class = "delta negative" if delta_week < 0 else "delta"
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Volume</h3>
            <div class='value'>{data['Volume'].iloc[-1]/1e6:.1f}M</div>
            <div class='{delta_class}'>Semaine: {delta_week:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        ma20 = data['Close'].tail(20).mean()
        delta_ma = ((current_price - ma20) / ma20 * 100)
        delta_class = "delta negative" if delta_ma < 0 else "delta"
        st.markdown(f"""
        <div class='metric-card'>
            <h3>MM20</h3>
            <div class='value'>${ma20:.2f}</div>
            <div class='{delta_class}'>vs prix: {delta_ma:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        delta_month = ((current_price - month_ago) / month_ago * 100)
        delta_class = "delta negative" if delta_month < 0 else "delta"
        st.markdown(f"""
        <div class='metric-card'>
            <h3>Performance</h3>
            <div class='value'>{delta_month:+.2f}%</div>
            <div class='{delta_class}'>30 jours</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Graphique
    st.subheader("📈 Aperçu historique")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data['Date'].tail(180),
        open=data['Open'].tail(180),
        high=data['High'].tail(180),
        low=data['Low'].tail(180),
        close=data['Close'].tail(180),
        name='AAPL'
    ))
    fig.update_layout(height=400, template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# PAGE DE PRÉDICTIONS
elif st.session_state.page == 'Prédictions':
    st.markdown("""
    <div class='main-header'>
        <h1>📊 Prédictions LSTM</h1>
        <p>Réseau de neurones récurrent</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_button = st.button("🚀 LANCER LA PRÉDICTION LSTM", type="primary", use_container_width=True)
    
    if predict_button and model:
        with st.spinner(f"Calcul des prédictions LSTM sur {jours} jours..."):
            
            # Préparer les données
            scaled_prices, scaler = prepare_data_for_lstm(data, window_size)
            
            # Prendre la dernière séquence
            last_sequence = scaled_prices[-window_size:].flatten()
            
            # Prédire
            predictions = predict_future(model, last_sequence, scaler, jours)
            
            # Créer les dates futures
            last_date = pd.to_datetime(data['Date'].iloc[-1])
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=jours, freq='D')
            
            # DataFrame des prédictions
            pred_df = pd.DataFrame({
                'ds': future_dates,
                'yhat1': predictions
            })
            
            # Graphique
            fig = go.Figure()
            
            # Historique
            hist_dates = pd.to_datetime(data['Date'].tail(90))
            fig.add_trace(go.Scatter(
                x=hist_dates,
                y=data['Close'].tail(90),
                mode='lines',
                name='Historique',
                line=dict(color='#1E3C72', width=2),
                fill='tozeroy',
                fillcolor='rgba(30,60,114,0.1)'
            ))
            
            # Prédictions LSTM
            fig.add_trace(go.Scatter(
                x=pred_df['ds'],
                y=pred_df['yhat1'],
                mode='lines+markers',
                name='Prédictions LSTM',
                line=dict(color='#DC143C', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond'),
                fill='tozeroy',
                fillcolor='rgba(220,20,60,0.1)'
            ))
            
            fig.add_vline(x=hist_dates.iloc[-1], line_dash="dash", line_color="gray")
            fig.update_layout(height=500, hovermode='x unified', template='plotly_white')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistiques
            st.subheader("📊 Statistiques")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prix min", f"${pred_df['yhat1'].min():.2f}")
            with col2:
                st.metric("Prix max", f"${pred_df['yhat1'].max():.2f}")
            with col3:
                st.metric("Prix moyen", f"${pred_df['yhat1'].mean():.2f}")
            with col4:
                trend = pred_df['yhat1'].iloc[-1] - pred_df['yhat1'].iloc[0]
                st.metric("Tendance", f"${trend:+.2f}")
            
            # Tableau
            st.subheader("📋 Détail")
            resultat = pd.DataFrame({
                'Date': pred_df['ds'].dt.strftime('%Y-%m-%d'),
                'Prix ($)': pred_df['yhat1'].round(2),
                'Variation ($)': pred_df['yhat1'].diff().round(2),
                'Variation (%)': (pred_df['yhat1'].pct_change() * 100).round(2)
            }).fillna(0)
            
            styled = resultat.style.format({
                'Prix ($)': '${:.2f}',
                'Variation ($)': '${:.2f}',
                'Variation (%)': '{:.2f}%'
            }).background_gradient(subset=['Prix ($)'], cmap='RdYlGn')
            
            st.dataframe(styled, width="stretch")

            
            # Téléchargement
            csv = resultat.to_csv(index=False)
            st.download_button("📥 Télécharger CSV", csv, f"lstm_pred_{datetime.now():%Y%m%d_%H%M}.csv")

# PAGE INFORMATIONS
elif st.session_state.page == 'Infos':
    st.markdown("""
    <div class='main-header'>
        <h1>ℹ️ À propos</h1>
        <p>Modèle LSTM</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='info-box'>
            <h3>🎯 LSTM</h3>
            <p>Long Short-Term Memory - Réseau de neurones récurrent</p>
            <h3>🔧 Architecture</h3>
            <ul>
                <li>Couches LSTM</li>
                <li>Dropout pour régularisation</li>
                <li>Couche Dense finale</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🍎 Apple LSTM Predictor</p>
</div>

""", unsafe_allow_html=True)

