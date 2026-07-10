import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import re
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai


def _betacf(a, b, x):
    # Continued-fraction evaluation for the incomplete beta function.
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    # Regularized incomplete beta function I_x(a, b).
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def pearson_p_value(r, n):
    # Two-sided p-value for a Pearson correlation via the Student's t
    # distribution — equivalent to scipy.stats.pearsonr, no scipy needed.
    if n <= 2 or pd.isna(r):
        return float('nan')
    if abs(r) >= 1.0:
        return 0.0
    df = n - 2
    t_sq = (r * r) * df / (1.0 - r * r)
    return _betai(0.5 * df, 0.5, df / (df + t_sq))

# Page configuration initialized first
st.set_page_config(page_title="QuantHelper | Asset Correlation", layout="wide")


# Data-grounded AI interpreter. It is fed ONLY the statistics we already
# computed and is instructed to interpret the data (not invent causes),
# with explicit statistical caveats. Cached so reruns don't burn quota.
@st.cache_data(show_spinner=False, ttl=3600)
def resolve_model(api_key):
    # Auto-detect a currently available generateContent model so the app
    # doesn't break each time Google retires a model name. Prefer the
    # cheapest flash-lite tier, then fall back gracefully.
    genai.configure(api_key=api_key)
    available = []
    for m in genai.list_models():
        methods = getattr(m, "supported_generation_methods", []) or []
        if "generateContent" in methods:
            available.append(m.name.replace("models/", ""))
    preferences = [
        "gemini-3.1-flash-lite", "gemini-3-flash", "gemini-3.5-flash",
        "gemini-flash-latest", "gemini-2.5-flash",
    ]
    for p in preferences:
        if p in available:
            return p
    for a in available:
        if "flash" in a and "lite" in a:
            return a
    for a in available:
        if "flash" in a:
            return a
    if available:
        return available[0]
    raise RuntimeError("No generateContent-capable model is available for this API key.")


@st.cache_data(show_spinner=False, ttl=3600)
def generate_ai_analysis(api_key, asset_a, asset_b, corr, p_value, n_obs,
                         roll_latest, roll_min, roll_max, roll_window):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(resolve_model(api_key))
    sig = "statistically significant" if p_value < 0.05 else "NOT statistically significant"
    prompt = f"""You are a careful quantitative analyst. You are given ONLY the computed statistics below,
describing the DAILY RETURN relationship between two assets over a lookback window.

Your job is to INTERPRET what these numbers show. Follow these rules strictly:
- Do NOT invent macroeconomic causes, news, or any external facts not present in the data.
- Do NOT claim causation — this is correlation of returns only.
- Be precise and plain. No hype, no filler.

Computed statistics:
- Asset A: {asset_a}
- Asset B: {asset_b}
- Daily return observations (n): {n_obs}
- Return correlation (Pearson r): {corr:.3f}
- Two-sided p-value: {p_value:.4f} ({sig})
- Rolling {roll_window}-day correlation — latest: {roll_latest:.2f}, min: {roll_min:.2f}, max: {roll_max:.2f}

Write exactly 3 short points, each on its own line. Do NOT number them and do NOT add bullet
characters — just plain text, one point per line:
- The strength and direction of the correlation, and whether it is significant given n.
- How stable the relationship has been, based on the rolling min/max/latest range.
- One concrete statistical caveat a reader should keep in mind.
"""
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

st.title("QuantHelper: Return Correlation Explorer")
st.markdown("<p style='color: #B2B5BE; margin-top: -15px;'>Correlation of daily returns across commodities, indices & crypto — with significance testing, rolling stability, and grounded AI interpretation. Built with Python, Streamlit & Gemini.</p>", unsafe_allow_html=True)

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
            # 4. Calculate correlation on DAILY RETURNS (not price levels).
            # Correlating raw price levels of two trending assets produces
            # spurious correlation; returns are the statistically correct input.
            returns = df_filtered[['Asset1', 'Asset2']].pct_change().dropna()
            n_obs = len(returns)

            if n_obs < 3 or returns['Asset1'].std() == 0 or returns['Asset2'].std() == 0:
                correlation_value, p_value = float('nan'), float('nan')
            else:
                correlation_value = float(returns['Asset1'].corr(returns['Asset2']))
                p_value = pearson_p_value(correlation_value, n_obs)

            is_significant = (not pd.isna(p_value)) and p_value < 0.05

            # Upper Display Row Block
            col1, col2 = st.columns([1.2, 2.8])
            with col1:
                val_display = f"{correlation_value:.2f}" if not pd.isna(correlation_value) else "N/A"
                st.metric(label="Return Correlation (Pearson r)", value=val_display)

                if pd.isna(correlation_value):
                    st.markdown("<p style='color: #B2B5BE; margin-top: 8px;'>Insufficient variance to compute</p>", unsafe_allow_html=True)
                elif correlation_value > 0.7:
                    st.markdown("<p style='color: #00E676; font-weight: 600; margin-top: 8px;'>● Strong Positive Co-movement</p>", unsafe_allow_html=True)
                elif correlation_value < -0.7:
                    st.markdown("<p style='color: #FF5252; font-weight: 600; margin-top: 8px;'>● Strong Inverse Co-movement</p>", unsafe_allow_html=True)
                else:
                    st.markdown("<p style='color: #78909C; margin-top: 8px;'>● Weak / Independent</p>", unsafe_allow_html=True)

                # Statistical significance + sample-size honesty
                if not pd.isna(p_value):
                    if is_significant:
                        st.markdown(f"<p style='color: #B2B5BE; font-size: 0.8rem; margin-top: 4px;'>Significant (p = {p_value:.3f}) · n = {n_obs} days</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color: #FFB74D; font-size: 0.8rem; margin-top: 4px;'>Not significant (p = {p_value:.3f}) · n = {n_obs} days</p>", unsafe_allow_html=True)
                if n_obs < 30:
                    st.markdown("<p style='color: #FFB74D; font-size: 0.8rem;'>⚠️ Small sample — widen the window for a reliable estimate.</p>", unsafe_allow_html=True)
                    
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
            
            chart_tab1, chart_tab2, chart_tab3 = st.tabs([
                "📈 Cumulative Performance",
                "🎯 Return Scatter & Regression",
                "🌊 Rolling Correlation",
            ])
            
            with chart_tab1:
                fig1 = px.line(df_normalized, y=[asset_1_label, asset_2_label], 
                              color_discrete_sequence=["#29B6F6", "#FF9100"], 
                              labels={"value": "% Cumulative Return", "Date": ""},
                              title=f"Cumulative Performance ({len(df_filtered)} Intersecting Sessions)")
                
                fig1.update_layout(hovermode="x unified", **global_chart_layout)
                fig1.update_xaxes(gridcolor='#232733', linecolor='#2A2E39')
                fig1.update_yaxes(gridcolor='#232733', linecolor='#2A2E39')
                st.plotly_chart(fig1, width="stretch")
                st.caption("Cumulative % return rebased to the window start. Shown for context — correlation is measured on daily returns, not these levels.")
                
            with chart_tab2:
                # Scatter of DAILY RETURNS — the correct visual for correlation.
                # Build from a DataFrame (proven-stable plotly path) rather than
                # passing raw arrays to px.scatter.
                x_col = f'{asset_1_label} Daily Return (%)'
                y_col = f'{asset_2_label} Daily Return (%)'
                scatter_df = pd.DataFrame({
                    x_col: returns['Asset1'].values * 100,
                    y_col: returns['Asset2'].values * 100,
                })
                
                fig2 = px.scatter(scatter_df, x=x_col, y=y_col,
                                  color_discrete_sequence=["#29B6F6"],
                                  title="Daily Return Distribution & OLS Fit")
                
                # OLS trendline over the return cloud
                x_data = scatter_df[x_col].values
                y_data = scatter_df[y_col].values
                if len(x_data) >= 2 and np.std(x_data) > 0:
                    slope, intercept = np.polyfit(x_data, y_data, 1)
                    x_trend = np.array([x_data.min(), x_data.max()])
                    y_trend = slope * x_trend + intercept
                    fig2.add_trace(go.Scatter(x=x_trend, y=y_trend, name='OLS Regression', 
                                              line=dict(color='#FF5252', width=2)))
                
                fig2.update_layout(**global_chart_layout)
                fig2.update_xaxes(gridcolor='#232733', linecolor='#2A2E39')
                fig2.update_yaxes(gridcolor='#232733', linecolor='#2A2E39')
                st.plotly_chart(fig2, width="stretch")
                st.caption("Each point is one trading day. A tight upward/downward line indicates strong positive/negative return correlation.")

            with chart_tab3:
                # Rolling correlation exposes regime shifts a single number hides.
                # Computed with a manual window loop over Series.corr (stable path)
                # instead of rolling().corr().
                roll_window = int(min(30, max(5, n_obs // 3)))
                roll_col = f'{roll_window}-Day Rolling r'
                a1 = returns['Asset1']
                a2 = returns['Asset2']
                roll_vals = []
                for end in range(roll_window, n_obs + 1):
                    seg1 = a1.iloc[end - roll_window:end]
                    seg2 = a2.iloc[end - roll_window:end]
                    roll_vals.append(seg1.corr(seg2))
                roll_index = returns.index[roll_window - 1:]
                rolling_df = pd.DataFrame({roll_col: roll_vals}, index=roll_index).dropna()

                if rolling_df.empty:
                    st.info("Not enough overlapping data to compute a rolling correlation. Widen the window.")
                else:
                    fig3 = px.line(rolling_df, y=roll_col,
                                   color_discrete_sequence=["#29B6F6"],
                                   title=f"{roll_window}-Day Rolling Return Correlation")
                    fig3.add_hline(y=0, line_dash="dash", line_color="#78909C")
                    fig3.update_yaxes(range=[-1, 1], gridcolor='#232733', linecolor='#2A2E39')
                    fig3.update_xaxes(gridcolor='#232733', linecolor='#2A2E39')
                    fig3.update_layout(hovermode="x unified", showlegend=False, **global_chart_layout)
                    st.plotly_chart(fig3, width="stretch")
                    st.caption("Correlation is not static — it shifts across market regimes. A flat single number can hide this instability.")
            
            # 6. Technical Methodology Block
            st.write("---")
            st.subheader("📋 Methodology & Computational Specifications")
            
            sig_txt = "not enough data" if pd.isna(p_value) else (
                f"significant (p = {p_value:.3f})" if is_significant else f"not significant (p = {p_value:.3f})")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.markdown(f"""
                **Data & Source**
                * Source: Yahoo Finance (daily EOD).
                * Prices: Adjusted close (splits/dividends).
                * Alignment: Inner join on shared trading sessions.
                """)
            with m_col2:
                st.markdown(f"""
                **Method**
                * Input: **Daily returns**, not price levels — avoids spurious correlation from shared trends.
                * Window: last `{len(df_filtered)}` trading sessions → `{n_obs}` return observations.
                * Rolling view: `{roll_window}`-session window to expose regime shifts.
                * Significance: two-sided p-value on Pearson $r$ — currently {sig_txt}.
                """)
            with m_col3:
                st.markdown("""
                **Correlation of returns**
                Pearson $r$ computed on daily percentage returns $r_{i,t}$:
                """)
                st.latex(r"\rho = \frac{\sum (r_{1,t} - \bar{r}_1)(r_{2,t} - \bar{r}_2)}{\sqrt{\sum (r_{1,t} - \bar{r}_1)^2 \sum (r_{2,t} - \bar{r}_2)^2}}")

            # 7. Data-Grounded AI Interpretation
            st.write("---")
            st.subheader("🤖 AI Data Interpretation")
            st.caption("Gemini interprets the computed statistics above — it is given only the numbers, "
                       "not asked to invent causes. Correlation is not causation.")

            has_rolling = ('rolling_df' in dir()) and (not rolling_df.empty)
            if pd.isna(correlation_value) or pd.isna(p_value) or not has_rolling:
                st.info("Not enough data to generate a grounded interpretation for this pair/window.")
            elif hasattr(st, "secrets") and "GENAI_API_KEY" in st.secrets:
                try:
                    with st.spinner("Interpreting the statistics..."):
                        analysis_text = generate_ai_analysis(
                            st.secrets["GENAI_API_KEY"],
                            asset_1_label, asset_2_label,
                            float(correlation_value), float(p_value), int(n_obs),
                            float(rolling_df[roll_col].iloc[-1]),
                            float(rolling_df[roll_col].min()),
                            float(rolling_df[roll_col].max()),
                            int(roll_window),
                        )
                    # Split into individual points, stripping any numbering or
                    # bullet markers the model may add, then render one per line.
                    raw_points = re.split(r'\n+|(?<![\d.])\d{1,2}[\).]\s+', analysis_text.strip())
                    points = [re.sub(r'^[\-\u2022\s]+', '', p).strip() for p in raw_points]
                    points = [p for p in points if p]
                    st.markdown("\n".join(f"- {p}" for p in points))
                except Exception as ai_err:
                    if "429" in str(ai_err) or "quota" in str(ai_err).lower():
                        st.warning("⏳ AI quota reached for now. Interpretation will return once quota resets.")
                    else:
                        st.warning("AI interpretation is temporarily unavailable.")
            else:
                st.info("💡 AI interpretation is offline (no API key configured). The statistics above are complete on their own.")

except Exception as e:
    st.error(f"Critical System Interruption: {e}")
