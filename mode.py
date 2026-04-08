import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
from pykalman import KalmanFilter
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# Popular stock tickers by category
STOCK_CATEGORIES = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM", "ORCL"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "LLY", "ABT", "DHR", "BMY"],
    "Consumer": ["AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "COST", "PG", "KO"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "ETFs": ["SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "TLT", "EEM", "VEA"]
}

# --- 1. DATA FETCHING & PREPROCESSING ---
def fetch_and_preprocess(ticker, period="3y"):
    """
    Downloads historical stock data and calculates technical indicators.
    
    What this does:
    - Fetches 3 years of daily price data
    - Calculates returns (daily % change)
    - Computes moving averages and RSI (momentum indicator)
    """
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False)
        
        if df.empty:
            return None

        # Fix MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure 'Adj Close' exists
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
            
        df['Return'] = df['Adj Close'].pct_change()
        
        # Technical Indicators
        df['MA20'] = df['Adj Close'].rolling(20).mean()
        df['MA50'] = df['Adj Close'].rolling(50).mean()
        
        # RSI Calculation
        delta = df['Adj Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Mid'] = df['Adj Close'].rolling(20).mean()
        df['BB_Std'] = df['Adj Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# --- 2. QUANTITATIVE MODELS ---
def run_quant_suite(df):
    """
    Runs multiple quantitative models to predict returns and measure risk.
    
    Models used:
    1. Linear Regression - Simple trend extrapolation
    2. ARIMA - Time series forecasting with memory
    3. GARCH - Volatility prediction (measures risk)
    4. Hidden Markov Model - Identifies market regime (bull/bear)
    5. Kalman Filter - Smooths price trends, removes noise
    """
    returns = df['Return'].values.reshape(-1, 1)
    prices = df['Adj Close'].values
    
    # A. Linear Regression (Simple Trend)
    X_lr = np.arange(len(df)).reshape(-1, 1)
    lr_model = LinearRegression().fit(X_lr[-60:], returns[-60:])
    r_hat_lr = lr_model.predict([[len(df)]])[0][0]

    # B. ARIMA (Time Series Forecast)
    try:
        arima_model = ARIMA(returns, order=(1, 1, 1)).fit()
        r_hat_arima = arima_model.forecast(steps=1)[0]
    except:
        r_hat_arima = r_hat_lr

    # C. GARCH (Volatility Forecast)
    garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1, dist='Normal')
    garch_res = garch_model.fit(disp='off')
    sigma_t = np.sqrt(garch_res.forecast(horizon=1).variance.values[-1, 0]) / 100

    # D. Hidden Markov Model (Market Regime)
    hmm = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    hmm.fit(returns)
    probs = hmm.predict_proba(returns)
    bull_idx = np.argmax(hmm.means_) 
    prob_bull = probs[-1, bull_idx]
    
    # Get regime history
    regime_states = hmm.predict(returns)

    # E. Kalman Filter (Trend Estimation)
    kf = KalmanFilter(
        transition_matrices=[1], 
        observation_matrices=[1], 
        initial_state_mean=prices[0], 
        initial_state_covariance=1, 
        observation_covariance=1, 
        transition_covariance=0.01
    )
    state_means, _ = kf.filter(prices)
    smoothed_prices = state_means.flatten()

    return r_hat_lr, r_hat_arima, sigma_t, prob_bull, smoothed_prices, regime_states, hmm

# --- 3. MONTE CARLO SIMULATION ---
def run_monte_carlo(current_price, sigma, mu, days=5, sims=1000):
    """
    Simulates possible future price paths using random sampling.
    
    Helps answer: "Where might the stock price be in 5 days?"
    - VaR (Value at Risk): Worst-case scenario (5th percentile)
    - Median: Most likely outcome
    - 95% Confidence: Best-case scenario
    """
    dt = 1  # Daily timestep
    returns_sim = np.random.normal(mu * dt, sigma * np.sqrt(dt), (sims, days))
    price_paths = current_price * np.exp(np.cumsum(returns_sim, axis=1))
    
    final_prices = price_paths[:, -1]
    var_5 = np.percentile(final_prices, 5)
    median = np.percentile(final_prices, 50)
    conf_95 = np.percentile(final_prices, 95)
    
    return var_5, median, conf_95, price_paths

# --- 4. DASHBOARD UI ---
st.set_page_config(
    page_title="Quantitative Trading Platform", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional Quantitative Trading Dashboard - For Educational Purposes Only"
    }
)

# Modern Professional CSS
st.markdown("""
<style>
    /* Import Modern Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #16213e 0%, #0f3460 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e8e8e8;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1600px;
    }
    
    /* Custom Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.25rem !important;
        margin-bottom: 1rem !important;
    }
    
    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #a0aec0 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
    }
    
    /* Card Styling */
    div[data-testid="stHorizontalBlock"] > div {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stHorizontalBlock"] > div:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.5);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        color: #e8e8e8 !important;
    }
    
    /* Dataframe Styling */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 500;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Custom Info Box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e8e8e8;
        backdrop-filter: blur(10px);
    }
    
    .info-box strong {
        color: #ffffff;
    }
    
    /* Warning Box */
    .stAlert {
        background: rgba(244, 114, 182, 0.1) !important;
        border-left: 4px solid #f472b6 !important;
        border-radius: 8px !important;
        color: #fce7f3 !important;
    }
    
    /* Success/Positive Text */
    .positive {
        color: #10b981 !important;
        font-weight: 600;
    }
    
    /* Negative Text */
    .negative {
        color: #ef4444 !important;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    }
    
    /* Selectbox and Input Styling */
    .stSelectbox > div > div, .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Slider Styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    /* Chart Container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Plotly Modebar */
    .modebar {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 8px !important;
        padding: 4px !important;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-bull {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    
    .badge-bear {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .badge-neutral {
        background: rgba(156, 163, 175, 0.2);
        color: #d1d5db;
        border: 1px solid rgba(156, 163, 175, 0.3);
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Text Colors */
    p, li, span {
        color: #d1d5db;
    }
    
    strong, b {
        color: #ffffff;
    }
    
    /* Code blocks */
    code {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        color: #fbbf24;
    }
</style>
""", unsafe_allow_html=True)

# Header with modern styling
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="margin-bottom: 0.5rem;">⚡ Quantitative Trading Platform</h1>
    <p style="font-size: 1.1rem; color: #a0aec0; font-weight: 400;">
        Advanced Multi-Model Analysis • Machine Learning • Risk Management
    </p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; font-size: 1.5rem; margin-bottom: 0;">⚙️ Configuration</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Stock Selection
    st.markdown("### 📈 Select Asset")
    category = st.selectbox("Category", list(STOCK_CATEGORIES.keys()), label_visibility="collapsed")
    ticker = st.selectbox("Ticker Symbol", STOCK_CATEGORIES[category], index=0)
    
    # Or custom ticker
    custom_ticker = st.text_input("Custom Ticker", "", placeholder="e.g., AAPL, TSLA")
    if custom_ticker:
        ticker = custom_ticker.upper()
    
    st.markdown("---")
    
    # Parameters
    st.markdown("### 🎯 Model Parameters")
    
    risk_k = st.slider(
        "Risk Scaling Factor", 
        0.01, 1.0, 0.2,
        help="Controls position sizing. Higher = more aggressive"
    )
    
    forecast_days = st.slider(
        "Forecast Horizon (Days)",
        1, 30, 5,
        help="Number of days to simulate future prices"
    )
    
    num_simulations = st.slider(
        "Monte Carlo Simulations",
        100, 5000, 1000, step=100,
        help="More simulations = more accurate probability estimates"
    )
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box" style="font-size: 0.85rem;">
        <strong>💡 Quick Guide</strong><br><br>
        1️⃣ Select your asset<br>
        2️⃣ Adjust risk parameters<br>
        3️⃣ Review signals & forecasts<br>
        4️⃣ Analyze Monte Carlo outcomes
    </div>
    """, unsafe_allow_html=True)

# --- MAIN CONTENT ---
df = fetch_and_preprocess(ticker)

if df is not None and len(df) > 100:
    # Get current stock info
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
    except:
        company_name = ticker
        sector = 'N/A'
    
    # Company Header
    st.markdown(f"""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.1);">
        <h2 style="margin: 0; font-size: 1.75rem;">{company_name}</h2>
        <p style="margin: 0.5rem 0 0 0; color: #a0aec0; font-size: 1rem;">
            {ticker} • {sector} • Last Updated: {df.index[-1].strftime("%B %d, %Y")} • {len(df):,} Data Points
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Run quantitative models
    r_lr, r_arima, sigma_t, prob_bull, kalman_prices, regime_states, hmm = run_quant_suite(df)
    
    current_price = df['Adj Close'].iloc[-1]
    
    # Calculate combined signal
    signal_base = (0.5 * r_lr) + (0.5 * r_arima)
    adj_signal = signal_base * (prob_bull / (sigma_t + 1e-9))
    
    # Position sizing
    pos_size = risk_k * (adj_signal / (sigma_t + 1e-9))
    pos_size = np.clip(pos_size, -1.0, 1.0)
    
    # Expected price tomorrow
    expected_return = signal_base
    expected_price = current_price * (1 + expected_return)
    price_change = expected_price - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # --- KEY METRICS ---
    st.markdown("## 📊 Live Metrics & Predictions")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:.2f}",
            help="Latest closing price"
        )
    
    with col2:
        st.metric(
            "Expected Tomorrow",
            f"${expected_price:.2f}",
            f"{price_change_pct:+.2f}%",
            delta_color="normal",
            help="Model prediction for next trading day"
        )
    
    with col3:
        regime_emoji = "🐂" if prob_bull > 0.5 else "🐻"
        regime_label = f"{regime_emoji} {'BULL' if prob_bull > 0.5 else 'BEAR'}"
        st.metric(
            "Market Regime",
            regime_label,
            f"{prob_bull:.1%}",
            delta_color="off",
            help="HMM model assessment of current market state"
        )
    
    with col4:
        st.metric(
            "Daily Volatility",
            f"{sigma_t*100:.2f}%",
            delta_color="off",
            help="GARCH forecast of tomorrow's price fluctuation"
        )
    
    with col5:
        position_label = "LONG" if pos_size > 0 else "SHORT" if pos_size < 0 else "NEUTRAL"
        st.metric(
            "Position Signal",
            position_label,
            f"{abs(pos_size)*100:.1f}%",
            delta_color="off",
            help="Suggested position based on signal strength and risk"
        )
    
    st.markdown("---")
    
    # --- EDUCATIONAL SECTION ---
    with st.expander("📚 Understanding the Metrics"):
        st.markdown("""
        **Current Price**: The most recent closing price of the stock.
        
        **Expected Tomorrow**: Our models' prediction for where the price will be in the next trading session. 
        This combines Linear Regression and ARIMA forecasts.
        
        **Market Regime**: Uses a Hidden Markov Model to determine if we're in a bullish (rising) or bearish (falling) market phase.
        - 🐂 Bull Market: Prices tend to rise, positive momentum
        - 🐻 Bear Market: Prices tend to fall, negative momentum
        
        **Daily Volatility**: Predicted by GARCH model. Higher volatility = more risk and larger price swings.
        - Low (<1%): Stable, predictable movement
        - Medium (1-3%): Normal market conditions
        - High (>3%): Increased risk and opportunity
        
        **Position Signal**: Based on signal strength and volatility:
        - LONG: Buy/hold the stock (expecting price increase)
        - SHORT: Sell/avoid the stock (expecting price decrease)
        - NEUTRAL: No clear signal, stay on sidelines
        """)
    
    # --- CHARTS ---
    
    # 1. Price Chart with Technical Indicators
    st.markdown("## 📉 Technical Analysis")
    
    # Create dark-themed plotly chart
    fig_price = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Price Action & Technical Indicators', 'RSI Momentum'),
        vertical_spacing=0.08,
        shared_xaxes=True
    )
    
    # Price and moving averages
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['Adj Close'], name='Price', 
                   line=dict(color='#667eea', width=2.5)),
        row=1, col=1
    )
    
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['MA20'], name='MA20',
                   line=dict(color='#f59e0b', width=2, dash='dash')),
        row=1, col=1
    )
    
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['MA50'], name='MA50',
                   line=dict(color='#8b5cf6', width=2, dash='dash')),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                   line=dict(color='rgba(156, 163, 175, 0.3)', width=1), showlegend=False),
        row=1, col=1
    )
    
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                   line=dict(color='rgba(156, 163, 175, 0.3)', width=1), fill='tonexty', 
                   fillcolor='rgba(156, 163, 175, 0.05)', showlegend=False),
        row=1, col=1
    )
    
    # Kalman Filter
    fig_price.add_trace(
        go.Scatter(x=df.index, y=kalman_prices, name='Kalman Trend',
                   line=dict(color='#06b6d4', width=2.5, dash='dot')),
        row=1, col=1
    )
    
    # RSI
    fig_price.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                   line=dict(color='#ec4899', width=2.5)),
        row=2, col=1
    )
    
    # RSI reference lines
    fig_price.add_hline(y=70, line_dash="dash", line_color="rgba(239, 68, 68, 0.5)", 
                        line_width=1.5, row=2, col=1)
    fig_price.add_hline(y=30, line_dash="dash", line_color="rgba(16, 185, 129, 0.5)", 
                        line_width=1.5, row=2, col=1)
    
    fig_price.update_xaxes(title_text="Date", row=2, col=1, color='#a0aec0')
    fig_price.update_yaxes(title_text="Price ($)", row=1, col=1, color='#a0aec0')
    fig_price.update_yaxes(title_text="RSI", row=2, col=1, color='#a0aec0')
    
    fig_price.update_layout(
        height=700,
        hovermode='x unified',
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="right", 
            x=1,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0')
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a0aec0', size=12),
        title_font=dict(color='#ffffff', size=14)
    )
    
    fig_price.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)')
    fig_price.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)')
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    with st.expander("📖 Chart Guide"):
        st.markdown("""
        **Price & Moving Averages:**
        - **Purple line**: Actual stock price
        - **Orange line (MA20)**: 20-day moving average - short-term trend
        - **Purple line (MA50)**: 50-day moving average - medium-term trend
        - **Gray shaded area**: Bollinger Bands - price touching upper band may signal overbought; lower band may signal oversold
        - **Cyan dotted line**: Kalman Filter - AI-smoothed trend removing noise
        
        **RSI (Relative Strength Index):**
        - Measures momentum on a scale of 0-100
        - **Above 70**: Potentially overbought (may reverse down)
        - **Below 30**: Potentially oversold (may reverse up)
        - **50**: Neutral momentum
        """)
    
    st.markdown("---")
    
    # 2. Monte Carlo Simulation
    st.markdown("## 🎲 Monte Carlo Simulation & Risk Analysis")
    
    col_mc1, col_mc2 = st.columns([2, 1])
    
    with col_mc1:
        var5, median, conf95, paths = run_monte_carlo(
            current_price, sigma_t, signal_base, 
            days=forecast_days, sims=num_simulations
        )
        
        fig_mc = go.Figure()
        
        # Show sample paths
        sample_size = min(50, num_simulations)
        for i in range(sample_size):
            fig_mc.add_trace(
                go.Scatter(
                    y=paths[i], 
                    mode='lines',
                    line=dict(width=0.8, color='rgba(102, 126, 234, 0.15)'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
        
        # Add percentile lines
        percentiles = np.percentile(paths, [5, 50, 95], axis=0)
        
        fig_mc.add_trace(
            go.Scatter(
                y=percentiles[0], 
                name='5th Percentile (VaR)',
                line=dict(color='#ef4444', width=3)
            )
        )
        
        fig_mc.add_trace(
            go.Scatter(
                y=percentiles[1], 
                name='Median (50th)',
                line=dict(color='#10b981', width=3)
            )
        )
        
        fig_mc.add_trace(
            go.Scatter(
                y=percentiles[2], 
                name='95th Percentile',
                line=dict(color='#f59e0b', width=3)
            )
        )
        
        fig_mc.update_layout(
            title=f"{num_simulations:,} Simulated Price Paths ({forecast_days}-Day Horizon)",
            xaxis_title=f"Days from Today",
            yaxis_title="Price ($)",
            height=450,
            hovermode='x',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0', size=12),
            title_font=dict(color='#ffffff', size=14),
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a0aec0')
            )
        )
        
        fig_mc.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)', color='#a0aec0')
        fig_mc.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)', color='#a0aec0')
        
        st.plotly_chart(fig_mc, use_container_width=True)
    
    with col_mc2:
        st.markdown("### Probability Distribution")
        
        # Histogram of final prices
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=paths[:, -1],
                nbinsx=40,
                name='Final Price Distribution',
                marker_color='#667eea',
                opacity=0.8
            )
        )
        
        fig_hist.add_vline(x=var5, line_dash="dash", line_color="#ef4444", line_width=2,
                          annotation_text="VaR 5%", annotation_position="top")
        fig_hist.add_vline(x=median, line_dash="dash", line_color="#10b981", line_width=2,
                          annotation_text="Median", annotation_position="top")
        fig_hist.add_vline(x=conf95, line_dash="dash", line_color="#f59e0b", line_width=2,
                          annotation_text="95%", annotation_position="top")
        
        fig_hist.update_layout(
            title="Final Price Distribution",
            xaxis_title="Price ($)",
            yaxis_title="Frequency",
            height=450,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0', size=12),
            title_font=dict(color='#ffffff', size=14)
        )
        
        fig_hist.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)', color='#a0aec0')
        fig_hist.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)', color='#a0aec0')
        
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Stats
        st.markdown("### Forecast Summary")
        st.markdown(f"""
        <div class="info-box">
        <strong>Value at Risk (5%):</strong><br/>
        ${var5:.2f} <span class="negative">({((var5/current_price - 1)*100):.1f}%)</span><br/><br/>
        
        <strong>Most Likely (Median):</strong><br/>
        ${median:.2f} <span class="{'positive' if median > current_price else 'negative'}">
        ({((median/current_price - 1)*100):+.1f}%)</span><br/><br/>
        
        <strong>Upside (95%):</strong><br/>
        ${conf95:.2f} <span class="positive">({((conf95/current_price - 1)*100):+.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📖 Understanding Monte Carlo"):
        st.markdown(f"""
        This simulation runs {num_simulations:,} possible future scenarios based on the stock's volatility 
        and expected return. Think of it as a "what if" analysis.
        
        **Value at Risk (VaR 5%)**: There's a 95% chance the price will be ABOVE this level. 
        This represents a worst-case scenario (only 5% of simulations go lower).
        
        **Median (50th percentile)**: The middle outcome - half of simulations are above, half below. 
        This is the most likely price target.
        
        **95th Percentile**: There's only a 5% chance the price will be THIS high or higher. 
        This represents an optimistic scenario.
        
        The wider the range between VaR and 95%, the more uncertainty (higher risk) there is.
        """)
    
    st.markdown("---")
    
    # 3. Market Regime Analysis
    st.markdown("## 🔄 Market Regime Detection")
    
    col_reg1, col_reg2 = st.columns([2, 1])
    
    with col_reg1:
        # Create regime visualization
        fig_regime = go.Figure()
        
        # Color code by regime
        bull_regime = regime_states == np.argmax(hmm.means_)
        
        fig_regime.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Adj Close'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=['#10b981' if b else '#ef4444' for b in bull_regime],
                    symbol='circle',
                    line=dict(width=0)
                ),
                name='Price',
                text=['Bull' if b else 'Bear' for b in bull_regime],
                hovertemplate='%{text}<br>Price: $%{y:.2f}<br>%{x}<extra></extra>'
            )
        )
        
        fig_regime.update_layout(
            title="Historical Regime Classification (HMM)",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=450,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a0aec0', size=12),
            title_font=dict(color='#ffffff', size=14)
        )
        
        fig_regime.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)', color='#a0aec0')
        fig_regime.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(255,255,255,0.05)', color='#a0aec0')
        
        st.plotly_chart(fig_regime, use_container_width=True)
    
    with col_reg2:
        st.markdown("### Current State")
        
        if prob_bull > 0.7:
            regime_desc = "Strong Bull 🐂"
            regime_gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
            regime_advice = "Market shows strong upward momentum. Consider maintaining or increasing long positions."
        elif prob_bull > 0.5:
            regime_desc = "Weak Bull 🐂"
            regime_gradient = "linear-gradient(135deg, #34d399 0%, #10b981 100%)"
            regime_advice = "Market is slightly bullish but uncertain. Use caution with new positions."
        elif prob_bull > 0.3:
            regime_desc = "Weak Bear 🐻"
            regime_gradient = "linear-gradient(135deg, #f87171 0%, #ef4444 100%)"
            regime_advice = "Market is slightly bearish. Consider reducing exposure or hedging."
        else:
            regime_desc = "Strong Bear 🐻"
            regime_gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"
            regime_advice = "Market shows strong downward pressure. Consider defensive positions."
        
        st.markdown(f"""
        <div style="background: {regime_gradient}; padding: 2rem; border-radius: 16px; text-align: center; margin-bottom: 1.5rem; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">{regime_desc}</h2>
            <h1 style="color: white; margin: 0.5rem 0 0 0; font-size: 3rem; font-weight: 700;">{prob_bull:.1%}</h1>
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-box">
        <strong>Interpretation:</strong><br/>
        {regime_advice}
        </div>
        """, unsafe_allow_html=True)
    
    with st.expander("📖 Regime Detection Explained"):
        st.markdown("""
        The Hidden Markov Model (HMM) identifies whether the market is in a "bull" or "bear" phase by analyzing 
        patterns in historical returns.
        
        **Bull Regime (Green)**: Period of positive momentum and rising prices
        **Bear Regime (Red)**: Period of negative momentum and falling prices
        
        The model automatically detects regime shifts without being explicitly told when they occur. 
        This helps traders:
        - Adjust risk based on market conditions
        - Avoid fighting the trend
        - Time entries and exits better
        
        Note: Past regimes don't guarantee future performance, but they provide valuable context.
        """)
    
    st.markdown("---")
    
    # 4. Model Breakdown
    st.markdown("## 🔬 Model Analytics")
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown("### Forecast Models")
        
        forecast_df = pd.DataFrame({
            "Model": ["Linear Regression", "ARIMA", "Combined Signal"],
            "Expected Return": [f"{r_lr*100:.4f}%", f"{r_arima*100:.4f}%", f"{signal_base*100:.4f}%"],
            "Description": [
                "Simple trend extrapolation",
                "Time series with memory",
                "Weighted average of models"
            ]
        })
        
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Risk Metrics")
        
        risk_df = pd.DataFrame({
            "Metric": ["GARCH Volatility", "Sharpe Ratio (Ann.)", "Max Drawdown"],
            "Value": [
                f"{sigma_t*100:.3f}%",
                f"{(df['Return'].mean() / df['Return'].std()) * np.sqrt(252):.2f}",
                f"{(df['Adj Close'] / df['Adj Close'].cummax() - 1).min()*100:.2f}%"
            ],
            "Description": [
                "Daily price fluctuation forecast",
                "Risk-adjusted return measure",
                "Largest peak-to-trough decline"
            ]
        })
        
        st.dataframe(risk_df, use_container_width=True, hide_index=True)
    
    with col_model2:
        st.markdown("### Trading Signal Breakdown")
        
        signal_df = pd.DataFrame({
            "Component": [
                "Base Signal",
                "Regime Adjustment",
                "Volatility Adjustment",
                "Final Adjusted Signal",
                "Position Size"
            ],
            "Value": [
                f"{signal_base:.6f}",
                f"×{prob_bull:.3f}",
                f"÷{sigma_t:.6f}",
                f"{adj_signal:.6f}",
                f"{pos_size*100:.2f}%"
            ],
            "Interpretation": [
                "Average of LR + ARIMA",
                "Scaled by bull probability",
                "Normalized by volatility",
                "Final trading signal",
                f"Recommended {'LONG' if pos_size > 0 else 'SHORT'} size"
            ]
        })
        
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Recent Performance")
        
        recent_returns = df['Return'].tail(30)
        perf_df = pd.DataFrame({
            "Period": ["1 Week", "2 Weeks", "1 Month"],
            "Return": [
                f"{df['Return'].tail(5).sum()*100:+.2f}%",
                f"{df['Return'].tail(10).sum()*100:+.2f}%",
                f"{df['Return'].tail(20).sum()*100:+.2f}%"
            ],
            "Volatility": [
                f"{df['Return'].tail(5).std()*100:.2f}%",
                f"{df['Return'].tail(10).std()*100:.2f}%",
                f"{df['Return'].tail(20).std()*100:.2f}%"
            ]
        })
        
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    # --- EDUCATIONAL FOOTER ---
    st.markdown("---")
    st.markdown("## 📚 Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["Model Glossary", "Risk Warnings", "Best Practices"])
    
    with tab1:
        st.markdown("""
        **Linear Regression**: Fits a straight line through recent returns to extrapolate the trend forward.
        - Pros: Simple, fast, good for stable trends
        - Cons: Assumes trends continue, bad at detecting reversals
        
        **ARIMA (AutoRegressive Integrated Moving Average)**: Time series model that uses past values and errors to predict future values.
        - Pros: Captures patterns, accounts for mean reversion
        - Cons: Computationally intensive, needs stationarity
        
        **GARCH (Generalized AutoRegressive Conditional Heteroskedasticity)**: Models volatility clustering - periods of high/low volatility.
        - Pros: Predicts risk accurately, captures volatility dynamics
        - Cons: Doesn't predict direction, only magnitude of moves
        
        **Hidden Markov Model**: Statistical model that identifies hidden states (bull/bear) from observable returns.
        - Pros: Detects regime changes, adapts to market conditions
        - Cons: Can be slow to react, assumes only 2 states
        
        **Kalman Filter**: Recursive algorithm that estimates true values by filtering out noise.
        - Pros: Excellent at smoothing, real-time updates
        - Cons: Assumes linear dynamics, needs parameter tuning
        """)
    
    with tab2:
        st.warning("""
        ⚠️ **IMPORTANT DISCLAIMERS**
        
        - This dashboard is for educational purposes only, not financial advice
        - Past performance does not guarantee future results
        - All models are simplifications of complex market dynamics
        - Quantitative models can and do fail, especially during:
            - Black swan events (2008 crisis, COVID-19)
            - Regime changes (policy shifts, technology disruption)
            - Low liquidity conditions
        - Always use proper risk management (stop losses, position sizing)
        - Never invest more than you can afford to lose
        - Consider consulting a licensed financial advisor
        """)
    
    with tab3:
        st.markdown("""
        **✅ Best Practices for Using Quantitative Models:**
        
        1. **Combine Multiple Models**: No single model is perfect. This dashboard uses 5+ models for robustness.
        
        2. **Understand the Assumptions**: Each model makes assumptions (stationarity, normality, etc.) that may not hold.
        
        3. **Use Proper Position Sizing**: The risk scaling factor (k) should reflect your risk tolerance and account size.
        
        4. **Monitor Regime Changes**: When market regime shifts, model performance can degrade. Watch the HMM signal.
        
        5. **Backtest Before Trading**: Test strategies on historical data before risking real capital.
        
        6. **Set Stop Losses**: Always have an exit plan if the trade goes against you.
        
        7. **Diversify**: Don't put all capital in one stock or strategy.
        
        8. **Stay Updated**: Markets evolve. Models need regular recalibration.
        
        9. **Paper Trade First**: Practice with virtual money before going live.
        
        10. **Keep Learning**: Quantitative finance is deep. This dashboard scratches the surface.
        """)

else:
    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 2rem; border-radius: 12px; margin: 2rem 0;">
        <h3 style="color: #ef4444; margin-top: 0;">❌ Unable to Fetch Data for {}</h3>
        <p style="color: #fca5a5;">
        <strong>Possible reasons:</strong><br>
        • Ticker symbol doesn't exist<br>
        • Insufficient historical data<br>
        • API connection issue
        </p>
        <p style="color: #fca5a5;">
        <strong>Try:</strong><br>
        • Verifying the ticker symbol<br>
        • Selecting a different stock from the sidebar<br>
        • Checking your internet connection
        </p>
    </div>
    """.format(ticker), unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0 1rem 0;'>
    <p style="font-size: 0.9rem; margin-bottom: 0.5rem;">Built with Streamlit • Data from Yahoo Finance</p>
    <p style="font-size: 0.85rem; margin: 0;">Models: scikit-learn, statsmodels, hmmlearn, arch</p>
    <p style="font-size: 0.75rem; margin-top: 1rem; color: #9ca3af;"><em>For educational purposes only. Not financial advice.</em></p>
</div>
""", unsafe_allow_html=True)