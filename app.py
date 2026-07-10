import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

# Page configuration initialized first
st.set_page_config(page_title="QuantHelper | Asset Correlation", layout="wide")


# Cache AI output so slider/rerun churn doesn't burn through API quota.
# Result is keyed by the asset pair and rounded correlation, so identical
# requests are served from cache instead of hitting the API again.
@st.cache_data(show_spinner=False, ttl=3600)
def generate_ai_analysis(api_key: str, asset_a: str, asset_b: str, corr: float) -> str:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = (f"Analyze the financial relationship between {asset_a} and {asset_b}. "
              f"The Pearson correlation coefficient over this period is {corr:.2f}. "
              f"Provide a brief 3-sentence macroeconomic explanation of why they might behave this way.")
    response = model.generate_content(prompt)
    return response.text

# FINANCIAL TERMINAL STYLE INJECTION (TradingView Charcoal Theme)
st.markdown("""
    <style>
        /* Eradicate the top white header leak and unify structural layers */
        html, body, .stApp, header[data-testid="stHeader"] { 
            background-color: #131722 !important; 
            color: #D1D4DC !important; 
        }
        
        /* Kill top decoration line entirely */
        div[data-testid="stDecoration"] { display: none !important; }
        
        /* Professional Trading View Sidebar Contrast Setup */
        [data-testid="stSidebar"] { 
            background-color: #171B26 !important; 
            border-right: 1px solid #2A2E39;
        }
        
        /* Crisp White Global Titles and Input Headings */
        h1, h2, h3, h4, h5, h6, label, .stMarkdown { 
            color: #FFFFFF !important; 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        
        /* Metric widget styling matching real quote boards */
        div[data-testid="stMetricValue"] { 
            color: #29B6F6 !important; 
            font-weight: 700; 
            font-size: 2.2rem !important;
        }
        div[data-testid="stMetricLabel"] { 
            color: #B2B5BE !important; 
            text-transform: uppercase;
            font-size: 0.75rem !important;
            letter-spacing: 0.5px;
        }
        
        /* Clean Terminal Tabs Interface Layout */
        button[data-baseweb="tab"] { 
            color: #B2B5BE !important; 
            background-color: transparent !important;
            border: none !important;
        }
        button[aria-selected="true"] { 
            color: #29B6F6 !important; 
            border-bottom: 2px solid #29B6F6 !important;
            font-weight: 600 !important;
        }
        
        /* Institutional Quote Cards Box Design */
        div[data-testid="stMetric"] {
            background-color: #171B26;
            padding: 15px;
            border-radius: 4px;
            border: 1px solid #2A2E39;
        }
    </style>
""", unsafe_allow_html=True)

st.title("QuantHelper: Quick Correlation Checker")
st.markdown("<p style='color: #B2B5BE; margin-top: -15px;'>Handshake AI Showcase Terminal MVP</p>", unsafe_allow_html=True)

# 1. Financial Asset Data Registry
ASSETS = {
    # --- Precious Metals ---
    "Gold (Commodity)": "GC=F",
    "Silver (Commodity)": "SI=F",
    "Platinum (Commodity)": "PL=F",
    "Palladium (Commodity)": "PA=F",
    "Copper (Commodity)": "HG=F",
    # --- Energy ---
    "Crude Oil WTI (Commodity)": "CL=F",
    "Crude Oil Brent (Commodity)": "BZ=F",
    "Natural Gas (Commodity)": "NG=F",
    "Gasoline RBOB (Commodity)": "RB=F",
    "Heating Oil (Commodity)": "HO=F",
    # --- Agriculture ---
    "Corn (Commodity)": "ZC=F",
    "Wheat (Commodity)": "ZW=F",
    "Soybeans (Commodity)": "ZS=F",
    "Coffee (Commodity)": "KC=F",
    "Sugar (Commodity)": "SB=F",
    "Cotton (Commodity)": "CT=F",
    "Cocoa (Commodity)": "CC=F",
    "Live Cattle (Commodity)": "LE=F",
    # --- Crypto ---
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    # --- Equities & Indices ---
    "S&P 500 Index": "^GSPC",
    "Nvidia (NVDA)": "NVDA",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Tesla (TSLA)": "TSLA",
    "Euro Stoxx 50 (Europe)": "^STOXX50E",
    "FTSE 100 (UK)": "^FTSE",
    "Nikkei 225 (Japan)": "^N225",
    "Hang Seng Index (Hong Kong)": "^HSI"
}

# 2. Sidebar Control Inputs
st.sidebar.markdown("<h3 style='margin-top: 0;'>Terminal Config</h3>", unsafe_allow_html=True)
asset_1_label = st.sidebar.selectbox("Select Asset 1", list(ASSETS.keys()), index=0)
asset_2_label = st.sidebar.selectbox("Select Asset 2", list(ASSETS.keys()), index=1)
days_window = st.sidebar.slider("Timeline Window (Days)", min_value=15, max_value=365, value=90)

ticker_1 = ASSETS[asset_1_label]
ticker_2 = ASSETS[asset_2_label]

# 3. Dynamic Data Sync Engine
@st.cache_data(ttl=3600)
def fetch_and_align_data(t1, t2, days):
    period = "2y"
    df1 = yf.Ticker(t1).history(period=period)
    df2 = yf.Ticker(t2).history(period=period)
    
    if df1.empty or df2.empty:
        return pd.DataFrame()
        
    df1 = df1[['Close']].rename(columns={'Close': 'Asset1'})
    df2 = df2[['Close']].rename(columns={'Close': 'Asset2'})
    
    df1.index = df1.index.tz_localize(None)
    df2.index = df2.index.tz_localize(None)
    
    merged = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
    return merged.tail(days)

try:
    df_filtered = fetch_and_align_data(ticker_1, ticker_2, days_window)
    
    if df_filtered.empty or len(df_filtered) < 5:
        st.error("⚠️ System Check Failed: Insufficient overlapping timeline points found.")
    else:
        # Check for zero value baseline anomaly protection
        base_val1 = df_filtered['Asset1'].iloc[0]
        base_val2 = df_filtered['Asset2'].iloc[0]
        
        if base_val1 == 0 or base_val2 == 0:
            st.error("⚠️ Outlier Alert: Base asset evaluation price cannot sit at zero.")
        else:
            # 4. Calculate Mathematical Correlation Coefficient
            correlation_value = df_filtered['Asset1'].corr(df_filtered['Asset2'])
            
            # Upper Display Row Block
            col1, col2 = st.columns([1.2, 2.8])
            with col1:
                val_display = f"{correlation_value:.2f}" if not pd.isna(correlation_value) else "0.00"
                st.metric(label="Pearson Correlation Score", value=val_display)
                
                if pd.isna(correlation_value):
                    st.markdown("<p style='color: #B2B5BE; margin-top: 8px;'>Stat Variance Gap</p>", unsafe_allow_html=True)
                elif correlation_value > 0.7:
                    st.markdown("<p style='color: #00E676; font-weight: 600; margin-top: 8px;'>● Strong Positive Alignment</p>", unsafe_allow_html=True)
                elif correlation_value < -0.7:
                    st.markdown("<p style='color: #FF5252; font-weight: 600; margin-top: 8px;'>● Strong Inverse Alignment</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #78909C; margin-top: 8px;'>● Independent / Weak Variance</p>", unsafe_allow_html=True)
                    
            # 5. Global Chart Workspace Configuration Definitions
            global_chart_layout = dict(
                plot_bgcolor='#171B26',
                paper_bgcolor='#131722',
                font_color='#D1D4DC',
                margin=dict(l=50, r=40, t=50, b=50)
            )
            
            # Formulate Clean Scaled Data Matrices
            df_normalized = pd.DataFrame(index=df_filtered.index)
            df_normalized[asset_1_label] = (df_filtered['Asset1'] / base_val1 - 1) * 100
            df_normalized[asset_2_label] = (df_filtered['Asset2'] / base_val2 - 1) * 100
            
            chart_tab1, chart_tab2 = st.tabs(["📈 Normalized Performance", "🎯 Statistical Scatter Regression"])
            
            with chart_tab1:
                fig1 = px.line(df_normalized, y=[asset_1_label, asset_2_label], 
                              color_discrete_sequence=["#29B6F6", "#FF9100"], 
                              labels={"value": "% Cumulative Return", "Date": ""},
                              title=f"Performance Trajectory ({len(df_filtered)} Intersecting Sessions)")
                
                fig1.update_layout(hovermode="x unified", **global_chart_layout)
                fig1.update_xaxes(gridcolor='#232733', linecolor='#2A2E39')
                fig1.update_yaxes(gridcolor='#232733', linecolor='#2A2E39')
                st.plotly_chart(fig1, width="stretch")
                
            with chart_tab2:
                # Build unified scale coordinate arrays to eliminate scale distortions
                x_data = df_normalized[asset_1_label].values
                y_data = df_normalized[asset_2_label].values
                
                # Zero-dependency manual OLS regression calculations using NumPy
                slope, intercept = np.polyfit(x_data, y_data, 1)
                x_trend = np.array([x_data.min(), x_data.max()])
                y_trend = slope * x_trend + intercept
                
                # Draw the scatter coordinate points
                fig2 = px.scatter(df_normalized, x=asset_1_label, y=asset_2_label,
                                  color_discrete_sequence=["#29B6F6"],
                                  labels={asset_1_label: f'{asset_1_label} Delta (%)', asset_2_label: f'{asset_2_label} Delta (%)'},
                                  title="Mathematical Asset Distribution & Normalized Return Trend")
                
                # Append manual trendline overlay trace cleanly
                fig2.add_trace(go.Scatter(x=x_trend, y=y_trend, name='OLS Regression', 
                                          line=dict(color='#FF5252', width=2)))
                
                fig2.update_layout(**global_chart_layout)
                fig2.update_xaxes(gridcolor='#232733', linecolor='#2A2E39')
                fig2.update_yaxes(gridcolor='#232733', linecolor='#2A2E39')
                st.plotly_chart(fig2, width="stretch")
            
            # 6. Technical Methodology Block
            st.write("---")
            st.subheader("📋 Methodology & Computational Specifications")
            
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                **Data Stream & Source**
                * Source: `Yahoo Finance API Engine`
                * Scope: Real-time adjusted closing prices.
                * Alignment: Clean cross-border intersection mapping.
                """)
            with m_col2:
                st.markdown(f"""
                **Temporal Parameters**
                * Window Size: `{days_window} Trading Days`
                * Sync Profile: Drops country holidays and weekends automatically via an inner-database join.
                """)
            with m_col3:
                st.markdown("""
                **Mathematical Engine**
                Calculated via the Pearson Product-Moment Correlation ($r$) over joint covariance partitions:
                """)
                st.latex(r"r = \frac{\sum (X - \bar{X})(Y - \bar{Y})}{\sqrt{\sum (X - \bar{X})^2 \sum (Y - \bar{Y})^2}}")

            # 7. Modern AI Core Engine Layer
            st.write("---")
            st.subheader("🤖 AI Macroeconomic Analysis")

            # Directly read from st.secrets without the restrictive if/else trap
            if hasattr(st, "secrets") and "GENAI_API_KEY" in st.secrets:
                try:
                    with st.spinner("Executing analytical processing..."):
                        analysis_text = generate_ai_analysis(
                            st.secrets["GENAI_API_KEY"],
                            asset_1_label,
                            asset_2_label,
                            round(correlation_value, 2),
                        )
                    st.markdown(f"<div style='color: #D1D4DC; line-height: 1.6;'>{analysis_text}</div>", unsafe_allow_html=True)
                except Exception as ai_err:
                    # Rate limit / quota exhaustion returns HTTP 429
                    if "429" in str(ai_err) or "quota" in str(ai_err).lower():
                        st.warning("⏳ AI engine is cooling down — the free-tier request quota was reached. "
                                   "Please wait a minute and rerun, or enable billing on your Google AI project for higher limits.")
                    else:
                        st.error(f"AI Engine Operational Exception: {ai_err}")
            else:
                st.info("💡 API Key Status: Missing 'GENAI_API_KEY' inside Streamlit Secrets panel.")

except Exception as e:
    st.error(f"Critical System Interruption: {e}")
