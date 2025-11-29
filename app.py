import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="Nakshatra Stock Analyst", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem; }
    .result-box { background-color: #f0f2f6; padding: 14px; border-radius: 8px; margin: 8px 0; }
    .prediction-positive { color: green; font-weight: bold; }
    .prediction-negative { color: red; font-weight: bold; }
    .prediction-neutral { color: orange; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class NakshatraStockAnalyzer:
    def __init__(self):
        # Fixed/normalized tickers
        self.script_mapping = {
            'NIFTY 50': '^NSEI',
            'BANK NIFTY': '^NSEBANK',
            'RELIANCE': 'RELIANCE.NS',
            'TCS': 'TCS.NS',
            'INFOSYS': 'INFY.NS',
            'HDFC BANK': 'HDFCBANK.NS',
            'TATA MOTORS': 'TATAMOTORS.NS',
            'WIPRO': 'WIPRO.NS',
            'ICICI BANK': 'ICICIBANK.NS',
            'SBI': 'SBIN.NS',
            'AXIS BANK': 'AXISBANK.NS'
        }
        self.nakshatras = [
            'Ashwini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashira', 'Ardra', 'Punarvasu',
            'Pushya', 'Ashlesha', 'Magha', 'Purva Phalguni', 'Uttara Phalguni', 'Hasta',
            'Chitra', 'Swati', 'Vishakha', 'Anuradha', 'Jyeshtha', 'Mula', 'Purva Ashadha',
            'Uttara Ashadha', 'Shravana', 'Dhanishta', 'Shatabhisha', 'Purva Bhadrapada',
            'Uttara Bhadrapada', 'Revati'
        ]

    @st.cache_data(show_spinner=False)
    def get_stock_data(self, script, period="6mo", interval="1d"):
        """
        Download stock data via yfinance with caching to prevent repeated network calls.
        Returns a DataFrame or None on failure.
        """
        ticker = self.script_mapping.get(script, script)
        try:
            data = yf.download(tickers=ticker, period=period, interval=interval, progress=False, threads=False)
            if data is None or data.empty:
                return None
            # Ensure required columns exist
            data = data.rename_axis('Date')
            return data
        except Exception as e:
            # don't raise inside app; return None so UI can show friendly message
            st.warning(f"Failed to download data for {ticker}: {e}")
            return None

    def simulate_nakshatra_data(self, date):
        # Accepts date or datetime.date
        date_obj = datetime.strptime(str(date)[:10], '%Y-%m-%d')
        day_of_year = date_obj.timetuple().tm_yday
        nakshatra_index = day_of_year % 27

        transitions = []
        if day_of_year % 3 == 0:
            transitions.append({'time': '10:30', 'from': self.nakshatras[nakshatra_index], 'to': self.nakshatras[(nakshatra_index + 1) % 27]})
        if day_of_year % 4 == 0:
            transitions.append({'time': '14:15', 'from': self.nakshatras[(nakshatra_index + 1) % 27], 'to': self.nakshatras[(nakshatra_index + 2) % 27]})

        return {
            'current_nakshatra': self.nakshatras[nakshatra_index],
            'nakshatra_index': nakshatra_index,
            'transitions': transitions,
            'moon_phase': 'Waxing' if day_of_year % 30 < 15 else 'Waning'
        }

    def generate_prediction(self, stock_data, nakshatra_data):
        if stock_data is None or len(stock_data) < 2:
            return {'direction': 'NEUTRAL ➡️', 'confidence': 50.0, 'key_times': [], 'reasoning': 'Insufficient data'}

        # Safer recent trend calculation: prefer 5-day pct change when available otherwise fallback
        try:
            pct5 = stock_data['Close'].pct_change(5).dropna()
            if len(pct5) > 0:
                recent_trend = float(pct5.iloc[-1])
            else:
                pct1 = stock_data['Close'].pct_change().dropna()
                recent_trend = float(pct1.iloc[-1]) if len(pct1) > 0 else 0.0
        except Exception:
            recent_trend = 0.0

        nakshatra_factor = np.sin(nakshatra_data['nakshatra_index'] * 2 * np.pi / 27)
        transition_factor = len(nakshatra_data['transitions']) * 0.12
        combined_score = (recent_trend * 0.45 + nakshatra_factor * 0.28 + transition_factor * 0.27)

        # Thresholds adjusted for more reasonable sensitivity
        if combined_score > 0.015:
            direction = "BULLISH 📈"
        elif combined_score < -0.015:
            direction = "BEARISH 📉"
        else:
            direction = "NEUTRAL ➡️"

        # Confidence: scale and clamp to [50, 90]
        raw_conf = (abs(combined_score) * 1000) + 50
        confidence = float(np.clip(raw_conf, 50, 90))

        key_times = []
        for transition in nakshatra_data['transitions']:
            impact = "Volatile" if nakshatra_data['nakshatra_index'] % 3 == 0 else "Moderate"
            key_times.append({
                'time': transition['time'],
                'expected_impact': impact,
                'period': f"{transition['from']} → {transition['to']}"
            })

        reasoning = f"Recent trend: {recent_trend:.2%}, Nakshatra: {nakshatra_data['current_nakshatra']}, Transitions: {len(nakshatra_data['transitions'])}"
        return {
            'direction': direction,
            'confidence': confidence,
            'key_times': key_times,
            'reasoning': reasoning
        }

def main():
    st.markdown('<div class="main-header">🌙 Nakshatra Stock Analyst</div>', unsafe_allow_html=True)
    analyzer = NakshatraStockAnalyzer()

    st.sidebar.header("Analysis Parameters")
    # Keep the select list in a stable order
    selected_script = st.sidebar.selectbox("Select Stock/Index:", sorted(list(analyzer.script_mapping.keys())))
    analysis_date = st.sidebar.date_input("Select Date:", datetime.now().date())

    # Allow period and interval selection
    period = st.sidebar.selectbox("History Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    interval = st.sidebar.selectbox("Data Interval:", ["1d", "1wk", "1mo"], index=0)

    if st.sidebar.button("Run Analysis", type="primary"):
        with st.spinner("Analyzing Nakshatra correlations..."):
            stock_data = analyzer.get_stock_data(selected_script, period=period, interval=interval)
            nakshatra_data = analyzer.simulate_nakshatra_data(analysis_date)
            prediction = analyzer.generate_prediction(stock_data, nakshatra_data)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 📊 Market Data")
                if stock_data is not None and not stock_data.empty:
                    latest = stock_data.iloc[-1]
                    st.metric("Current Price", f"₹{latest['Close']:.2f}")
                    daily_change = latest['Close'] - latest['Open']
                    st.metric("Daily Change", f"₹{daily_change:.2f}")
                    st.write(f"Data range: {stock_data.index[0].date()} → {stock_data.index[-1].date()}")
                else:
                    st.write("Data not available for the selected period/interval")

            with col2:
                st.markdown("### 🌙 Nakshatra Analysis")
                st.metric("Current Nakshatra", nakshatra_data['current_nakshatra'])
                st.metric("Moon Phase", nakshatra_data['moon_phase'])
                st.metric("Transitions Today", len(nakshatra_data['transitions']))

            with col3:
                st.markdown("### 🔮 Prediction")
                direction_class = "prediction-positive" if "BULLISH" in prediction['direction'] else "prediction-negative" if "BEARISH" in prediction['direction'] else "prediction-neutral"
                st.markdown(f'<div class="result-box {direction_class}">', unsafe_allow_html=True)
                st.metric("Direction", prediction['direction'])
                st.metric("Confidence", f"{prediction['confidence']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            col4, col5 = st.columns(2)

            with col4:
                st.markdown("### 📈 Price Chart")
                if stock_data is not None and not stock_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=stock_data.index,
                        open=stock_data['Open'],
                        high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        name="Price"
                    ))
                    fig.update_layout(title=f"{selected_script} Price Movement", height=400, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Chart data not available")

            with col5:
                st.markdown("### 🕒 Key Time Periods")
                if prediction['key_times']:
                    for period_info in prediction['key_times']:
                        st.markdown(f"""
                        <div class="result-box">
                        <h4>⏰ {period_info['time']}</h4>
                        <p><strong>Transition:</strong> {period_info['period']}</p>
                        <p><strong>Expected Impact:</strong> {period_info['expected_impact']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No significant Nakshatra transitions today")

                st.markdown("### 📋 Analysis Reasoning")
                st.info(prediction['reasoning'])

    else:
        st.info("👈 Select parameters and click 'Run Analysis' to begin!")
        st.markdown("""
        ### 🌟 Welcome to Nakshatra Stock Analyst!

        **Supported Stocks/Indices:**
        - NIFTY 50, BANK NIFTY
        - RELIANCE, TCS, INFOSYS
        - HDFC BANK, ICICI BANK, SBI, AXIS BANK

        **How it works:**
        1. Select a stock/index
        2. Choose analysis date and historical period
        3. Get Nakshatra-based predictions
        4. View key time periods

        *Note: This is for educational purposes only.*
        """)

if __name__ == "__main__":
    main()