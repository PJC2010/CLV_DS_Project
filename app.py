import pandas as pd
import streamlit as st
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
import plotly.graph_objects as go

# --- 1. Data Loading ---
@st.cache_data
def load_local_transaction_data():
    """
    Loads the CDNOW customer dataset from a local CSV file.
    """
    st.write("Loading transaction data from local cdnow.csv file...")
    try:
        df = pd.read_csv('cdnow.csv', index_col=0, parse_dates=['date'])
        df.rename(columns={'price': 'monetary_value'}, inplace=True)
        return df
    except FileNotFoundError:
        st.error("Error: 'cdnow.csv' not found. Please make sure the file is in the same directory as app.py.")
        return None

# --- 2. Lifetimes Data Preparation ---
def prepare_lifetimes_data(df):
    """
    Prepares the data into the Recency, Frequency, Monetary (RFM) format
    required by the lifetimes library.
    """
    observation_period_end = df['date'].max()
    summary_df = summary_data_from_transaction_data(
        df,
        customer_id_col='customer_id',
        datetime_col='date',
        monetary_value_col='monetary_value',
        observation_period_end=observation_period_end
    )
    return summary_df

# --- 3. Model Training ---
def train_clv_models(summary_df):
    """
    Trains the BG/NBD and Gamma-Gamma models.
    """
    ggf_ready_df = summary_df.query("frequency > 0 and monetary_value > 0")
    bgf = BetaGeoFitter(penalizer_coef=0.0)
    bgf.fit(summary_df['frequency'], summary_df['recency'], summary_df['T'])
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(ggf_ready_df['frequency'], ggf_ready_df['monetary_value'])
    return bgf, ggf

# --- 4. Streamlit App UI ---
st.set_page_config(layout="wide", page_title="CLV Prediction Engine")

st.title("📈 Customer Lifetime Value (CLV) Prediction Engine")
st.markdown("""
This tool predicts the **12-month Customer Lifetime Value (CLV)** using a local e-commerce dataset.
It uses the **BG/NBD model** to forecast future transactions and the **Gamma-Gamma model** to predict monetary value.
""")

# --- Main App Logic ---
transaction_df = load_local_transaction_data()

if transaction_df is not None:
    rfm_df = prepare_lifetimes_data(transaction_df)
    bgf_model, ggf_model = train_clv_models(rfm_df)

    # --- Section 1: Overall CLV Forecast ---
    st.header("Overall Customer Base CLV Forecast")
    with st.spinner("Calculating CLV for all customers..."):
        clv_forecast = ggf_model.customer_lifetime_value(
            bgf_model,
            rfm_df['frequency'],
            rfm_df['recency'],
            rfm_df['T'],
            rfm_df['monetary_value'],
            time=12,
            discount_rate=0.01
        )
        rfm_df['predicted_clv_12_months'] = clv_forecast

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rfm_df['predicted_clv_12_months'], nbinsx=50, name='CLV'))
    fig.update_layout(
        title_text='Distribution of Predicted 12-Month CLV Across All Customers',
        xaxis_title='Predicted CLV ($)',
        yaxis_title='Number of Customers',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Section 2: Individual Customer Analysis ---
    st.header("Individual Customer Analysis")
    customer_list = rfm_df.index.astype(str).tolist()
    selected_customer = st.selectbox("Select a Customer ID to Analyze:", customer_list)

    if selected_customer:
        customer_id_int = int(selected_customer)
        customer_data = rfm_df.loc[customer_id_int]
        customer_clv = customer_data['predicted_clv_12_months']

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Customer {customer_id_int} - Historical Data")
            st.metric("Frequency (Total Purchases)", f"{int(customer_data['frequency'])}")
            st.metric("Recency (Days Since Last Purchase)", f"{int(customer_data['recency'])}")
            st.metric("Average Monetary Value", f"${customer_data['monetary_value']:.2f}")
        with col2:
            st.subheader("Future Value Prediction")
            st.metric("Predicted 12-Month CLV", f"${customer_clv:.2f}")
            st.subheader("Strategic Recommendation")
            clv_threshold_high = rfm_df['predicted_clv_12_months'].quantile(0.8)
            clv_threshold_low = rfm_df['predicted_clv_12_months'].quantile(0.4)
            if customer_clv > clv_threshold_high:
                st.success("High-Value Customer: Target with VIP offers, loyalty programs, and personalized communication.")
            elif customer_clv > clv_threshold_low:
                st.info("Medium-Value Customer: Nurture with targeted email campaigns and re-engagement offers.")
            else:
                st.warning("Low-Value/At-Risk Customer: Include in general marketing but avoid high-cost acquisition/retention efforts.")

    # --- Section 3: Model Fit Assessment ---
    # MOVED SECTION: This is now outside the 'if selected_customer' block.
    # It will always be displayed on the dashboard.
    st.header("Model Fit Assessment")
    st.markdown("This plot compares the model's prediction of repeat purchases against the actual data. A good fit means the blue and orange lines are close.")
    
    fig_val = go.Figure()
    plot_period_transactions(bgf_model, fig=fig_val)
    st.plotly_chart(fig_val, use_container_width=True)

    # --- Section 4: Data Display ---
    with st.expander("Show Top 10 Customers by Predicted CLV"):
        st.dataframe(rfm_df.sort_values(by='predicted_clv_12_months', ascending=False).head(10))
