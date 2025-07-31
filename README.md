# Customer Lifetime Value (CLV) Prediction Engine

This project is a full-stack data science application that predicts the 12-month Customer Lifetime Value (CLV) for an e-commerce customer base. It moves beyond historical analysis to provide forward-looking, actionable insights for financial and marketing strategy. The entire application is deployed as an interactive web dashboard using Streamlit.

![Screenshot of the CLV Dashboard](https://imgur.com/a/4fpcECg)

## 1. The Business Problem

In any business, not all customers are created equal. Marketing and retention efforts are often expensive, and applying them uniformly is inefficient. While historical data analysis can identify who *was* a valuable customer, it fails to answer two critical questions for financial planning:

1.  **Future Value:** How much is a new or existing customer likely to be worth in the future?
2.  **Resource Allocation:** How should we allocate our marketing budget to maximize return on investment? Which customers should receive premium service, and which should be targeted with low-cost campaigns?

Answering these questions requires moving from descriptive analytics to predictive forecasting.

## 2. The Solution

This project solves the problem by creating a **CLV Prediction Engine**. It analyzes a customer's past transaction history and uses probabilistic models to forecast their future financial value.

The engine is deployed as an interactive dashboard that allows a non-technical user (e.g., a marketing manager) to:
* View the distribution of predicted CLV across the entire customer base.
* Drill down into any individual customer to see their historical data and their predicted 12-month CLV.
* Receive a clear, data-driven strategic recommendation for how to treat that customer (e.g., VIP, Nurture, Low-Priority).
* Assess the fit and performance of the underlying predictive model.

## 3. Tech Stack & Libraries

* **Language:** Python 3
* **Core Libraries:**
    * **Pandas:** For data manipulation and preparation.
    * **Lifetimes:** The core engine for CLV modeling, providing implementations of the BG/NBD and Gamma-Gamma models.
    * **Streamlit:** For building and deploying the interactive web dashboard.
    * **Plotly:** For creating rich, interactive data visualizations.

## 4. Methodology

The project follows a clear, end-to-end data science workflow:

#### Step 1: Data Loading and Preparation
The application loads raw transactional data from a local `cdnow.csv` file. This data contains customer IDs, transaction dates, and purchase values.

#### Step 2: RFM Transformation
The raw transaction log is transformed into a **Recency, Frequency, Monetary (RFM)** summary format using the `lifetimes` library. For each customer, we calculate:
* **Recency:** The time between the customer's first and last purchase.
* **Frequency:** The number of repeat purchases.
* **T:** The total "age" of the customer (time since their first purchase).
* **Monetary Value:** The average value of their repeat purchases.

#### Step 3: Probabilistic Modeling
Two distinct models are trained on the RFM data:

1.  **Beta-Geometric/Negative Binomial Distribution (BG/NBD) Model:** This model analyzes a customer's recency and frequency to predict their future purchasing behavior. It answers two questions:
    * Is this customer likely to still be "alive" or have they churned?
    * If they are alive, how many transactions will they make in a given future time period?

2.  **Gamma-Gamma Model:** This model analyzes the monetary value of a customer's past purchases to predict the value of their future purchases. It assumes there is no relationship between the frequency and the value of transactions. It answers the question:
    * When this customer makes a purchase, what is its likely monetary value?

#### Step 4: CLV Calculation & Deployment
The predictions from both models are combined to calculate the final 12-month CLV for each customer. The entire workflow is wrapped in a Streamlit application to create an accessible and interactive tool for end-users.

## 5. How to Run the Application

#### Prerequisites
* Python 3.7+
* The `cdnow.csv` file in the same directory as the application script.

#### Installation
1.  Clone the repository or download the source code.
2.  Install the required Python libraries:
    ```bash
    pip install pandas lifetimes streamlit plotly
    ```

#### Execution
1.  Navigate to the project directory in your terminal.
2.  Run the following command:
    ```bash
    streamlit run app.py
    ```
3.  Your default web browser will open with the interactive dashboard.