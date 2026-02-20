import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Single-Index Model Explorer", layout="wide")

st.title("📊 Single-Index Model Regression Explorer")
st.markdown("""
This interactive tool helps you understand how the single-index model works by letting you:
- Generate synthetic stock returns with different α, β, and firm-specific risk
- See the resulting scatter plot and regression line
- Understand regression statistics like R², t-statistics, and p-values
""")

# Sidebar for parameters
st.sidebar.header("📝 Model Parameters")
st.sidebar.markdown("Adjust these to see how they affect the regression:")

# User inputs
n_months = st.sidebar.slider("Number of Months", 24, 120, 60, 6)
alpha_true = st.sidebar.slider("True Alpha (monthly %)", -5.0, 5.0, 0.0, 0.5) / 100
beta_true = st.sidebar.slider("True Beta", 0.0, 3.0, 1.0, 0.1)
firm_risk = st.sidebar.slider("Firm-Specific Risk (monthly %)", 0.0, 30.0, 10.0, 1.0) / 100
market_risk = st.sidebar.slider("Market Risk (monthly %)", 2.0, 10.0, 5.0, 0.5) / 100

# Add a random seed slider for reproducibility
random_seed = st.sidebar.slider("Random Seed (for different samples)", 1, 100, 42, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips:")
st.sidebar.markdown("""
- **Increase β**: Watch how the regression line gets steeper
- **Increase firm-specific risk**: See points scatter more widely around the line (lower R²)
- **Set α ≠ 0**: Observe the intercept shift
- **Change random seed**: See different random samples with the same parameters
""")

# Generate data
np.random.seed(random_seed)

# Generate market returns (normally distributed)
market_returns = np.random.normal(0, market_risk, n_months)

# Generate firm-specific shocks
firm_shocks = np.random.normal(0, firm_risk, n_months)

# Generate stock returns using single-index model
stock_returns = alpha_true + beta_true * market_returns + firm_shocks

# Run regression
slope, intercept, r_value, p_value, std_err = stats.linregress(market_returns, stock_returns)

# Calculate additional statistics
r_squared = r_value ** 2
alpha_estimate = intercept
beta_estimate = slope

# Calculate standard errors and t-statistics
n = len(market_returns)
residuals = stock_returns - (alpha_estimate + beta_estimate * market_returns)
residual_std = np.std(residuals, ddof=2)

# Standard error of beta
se_beta = std_err
t_stat_beta = beta_estimate / se_beta
p_value_beta = 2 * (1 - stats.t.cdf(abs(t_stat_beta), n - 2))

# Standard error of alpha
x_mean = np.mean(market_returns)
se_alpha = residual_std * np.sqrt(1 / n + x_mean ** 2 / np.sum((market_returns - x_mean) ** 2))
t_stat_alpha = alpha_estimate / se_alpha
p_value_alpha = 2 * (1 - stats.t.cdf(abs(t_stat_alpha), n - 2))

# Systematic and firm-specific variance
systematic_var = (beta_estimate ** 2) * (np.var(market_returns, ddof=1))
firm_specific_var = residual_std ** 2
total_var = np.var(stock_returns, ddof=1)

# Create three columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scatter plot
    ax.scatter(market_returns * 100, stock_returns * 100, alpha=0.6, s=50, color='darkred')
    
    # Regression line
    x_line = np.array([market_returns.min(), market_returns.max()])
    y_line = alpha_estimate + beta_estimate * x_line
    ax.plot(x_line * 100, y_line * 100, 'b-', linewidth=2.5, label='Security Characteristic Line (SCL)')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # Labels
    ax.set_xlabel('Monthly Excess Return, Market Index (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Monthly Excess Return, Stock (%)', fontsize=12, fontweight='bold')
    ax.set_title('Security Characteristic Line (SCL)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    
    st.pyplot(fig)
    
    # Show equation
    st.markdown("### 📐 Estimated Regression Equation:")
    st.latex(f"R_{{stock}} = {alpha_estimate*100:.3f}\\% + {beta_estimate:.3f} \\times R_{{market}} + e")

with col2:
    st.markdown("### 📊 Regression Statistics")

    # Create a nice formatted statistics table
    st.markdown("#### Model Fit:")
    st.metric("R-squared", f"{r_squared:.4f}", help="Proportion of variance explained by the market")
    st.metric("Correlation", f"{r_value:.4f}", help="Correlation between stock and market")
    st.metric("Residual Std Dev", f"{residual_std * 100:.2f}%", help="Typical size of firm-specific shocks")

    st.markdown("---")
    st.markdown("#### Alpha (Intercept):")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.metric("Estimate", f"{alpha_estimate * 100:.3f}%")
        st.metric("Std Error", f"{se_alpha * 100:.3f}%")
    with col_a2:
        st.metric("t-statistic", f"{t_stat_alpha:.3f}")
        st.metric("p-value", f"{p_value_alpha:.3f}")

    # Interpretation
    if p_value_alpha < 0.05:
        st.success("✅ Alpha is statistically significant (p < 0.05)")
    else:
        st.warning("⚠️ Alpha is NOT statistically significant (p ≥ 0.05)")

    st.markdown("---")
    st.markdown("#### Beta (Slope):")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.metric("Estimate", f"{beta_estimate:.3f}")
        st.metric("Std Error", f"{se_beta:.3f}")
    with col_b2:
        st.metric("t-statistic", f"{t_stat_beta:.3f}")
        st.metric("p-value", f"{p_value_beta:.4f}")

    # Interpretation
    if p_value_beta < 0.05:
        st.success("✅ Beta is statistically significant (p < 0.05)")
        if beta_estimate > 1:
            st.info("📈 Aggressive stock (β > 1): Amplifies market movements")
        elif beta_estimate < 1:
            st.info("📉 Defensive stock (β < 1): Dampens market movements")
        else:
            st.info("➡️ Neutral stock (β ≈ 1): Moves with the market")
    else:
        st.warning("⚠️ Beta is NOT statistically significant (p ≥ 0.05)")

# Risk decomposition section
st.markdown("---")
st.markdown("## 🎯 Risk Decomposition")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("### Total Risk")
    st.metric("Variance", f"{total_var:.6f}")
    st.metric("Std Dev (monthly)", f"{np.sqrt(total_var) * 100:.2f}%")
    st.metric("Std Dev (annual)", f"{np.sqrt(total_var) * np.sqrt(12) * 100:.2f}%")

with col4:
    st.markdown("### Systematic Risk")
    st.metric("Variance", f"{systematic_var:.6f}")
    st.metric("Std Dev (monthly)", f"{np.sqrt(systematic_var) * 100:.2f}%")
    pct_systematic = (systematic_var / total_var) * 100 if total_var > 0 else 0
    st.metric("% of Total", f"{pct_systematic:.1f}%")

with col5:
    st.markdown("### Firm-Specific Risk")
    st.metric("Variance", f"{firm_specific_var:.6f}")
    st.metric("Std Dev (monthly)", f"{np.sqrt(firm_specific_var) * 100:.2f}%")
    pct_firm = (firm_specific_var / total_var) * 100 if total_var > 0 else 0
    st.metric("% of Total", f"{pct_firm:.1f}%")

# Visualization of risk decomposition
fig2, ax2 = plt.subplots(figsize=(10, 3))
risks = [pct_systematic, pct_firm]
labels = ['Systematic\n(Market)', 'Firm-Specific\n(Diversifiable)']
colors = ['#1f77b4', '#ff7f0e']

ax2.barh([0], [pct_systematic], color=colors[0], label='Systematic Risk')
ax2.barh([0], [pct_firm], left=[pct_systematic], color=colors[1], label='Firm-Specific Risk')
ax2.set_xlim(0, 100)
ax2.set_ylim(-0.5, 0.5)
ax2.set_xlabel('Percentage of Total Risk (%)', fontsize=12, fontweight='bold')
ax2.set_yticks([])
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=2, fontsize=11)
ax2.set_title('Risk Decomposition', fontsize=13, fontweight='bold', pad=20)

# Add percentage labels
ax2.text(pct_systematic / 2, 0, f'{pct_systematic:.1f}%',
         ha='center', va='center', fontsize=12, fontweight='bold', color='white')
ax2.text(pct_systematic + pct_firm / 2, 0, f'{pct_firm:.1f}%',
         ha='center', va='center', fontsize=12, fontweight='bold', color='white')

st.pyplot(fig2)

# Key insights section
st.markdown("---")
st.markdown("## 💡 Key Insights")

col6, col7 = st.columns(2)

with col6:
    st.markdown("### Understanding R²")
    st.markdown(f"""
    - R² = {r_squared:.1%} means that **{r_squared * 100:.1f}%** of the stock's return variation is explained by market movements
    - The remaining **{(1 - r_squared) * 100:.1f}%** is due to firm-specific factors
    - Higher R² means the stock moves more closely with the market
    """)

    st.markdown("### Understanding Statistical Significance")
    st.markdown(f"""
    - **Alpha p-value = {p_value_alpha:.3f}**: {"Not significant" if p_value_alpha >= 0.05 else "Significant"}
      - This tells us whether the stock has abnormal returns beyond what beta predicts
    - **Beta p-value = {p_value_beta:.4f}**: {"Not significant" if p_value_beta >= 0.05 else "Significant"}
      - This tells us whether the stock's sensitivity to the market is reliably different from zero
    """)

with col7:
    st.markdown("### Investment Implications")
    if beta_estimate > 1.5:
        st.markdown("🚀 **High Beta Stock**: Expect large swings in both directions when the market moves")
    elif beta_estimate > 1.0:
        st.markdown("📈 **Moderately Aggressive**: Amplifies market movements")
    elif beta_estimate > 0.5:
        st.markdown("➡️ **Moderate Risk**: Moves somewhat with the market")
    else:
        st.markdown("🛡️ **Defensive Stock**: Provides relative stability in volatile markets")

    if pct_firm > 70:
        st.markdown(
            f"⚠️ **High Firm-Specific Risk** ({pct_firm:.0f}%): This stock's returns are driven more by company events than market movements. Diversification is especially important!")
    elif pct_firm > 40:
        st.markdown(
            f"⚡ **Moderate Firm-Specific Risk** ({pct_firm:.0f}%): A mix of market and company factors drive returns")
    else:
        st.markdown(f"🎯 **Market-Driven Stock** ({pct_systematic:.0f}% systematic): Returns closely track the market")

# Download data option
st.markdown("---")
st.markdown("## 💾 Download Data")

# Create dataframe
df = pd.DataFrame({
    'Market_Return_%': market_returns * 100,
    'Stock_Return_%': stock_returns * 100,
    'Fitted_Value_%': (alpha_estimate + beta_estimate * market_returns) * 100,
    'Residual_%': residuals * 100
})

csv = df.to_csv(index=False)
st.download_button(
    label="📥 Download Data as CSV",
    data=csv,
    file_name="single_index_model_data.csv",
    mime="text/csv"
)

# Show sample data
with st.expander("📋 View Data Sample"):
    st.dataframe(df.head(10))

# Educational notes
st.markdown("---")
st.markdown("## 📚 Educational Notes")

with st.expander("🎓 What is the Single-Index Model?"):
    st.markdown("""
    The single-index model assumes that stock returns can be decomposed into:

    $$R_{stock} = \\alpha + \\beta R_{market} + e$$

    Where:
    - **α (alpha)**: The expected return when the market return is zero (intercept)
    - **β (beta)**: The stock's sensitivity to market movements (slope)
    - **e**: Firm-specific surprises (residual)

    This model simplifies portfolio analysis by reducing the number of parameters needed from n² to 3n+2.
    """)

with st.expander("📊 How to Interpret Regression Statistics"):
    st.markdown("""
    **R-squared (R²)**:
    - Ranges from 0 to 1
    - Measures the proportion of variance explained by the model
    - Higher values mean the stock tracks the market more closely

    **Standard Error**:
    - Measures the typical uncertainty in our estimates
    - Smaller standard errors mean more precise estimates
    - If standard error ≈ estimate size, the estimate is very uncertain

    **t-statistic**:
    - Calculated as: estimate / standard error
    - Values beyond ±2 are generally considered significant
    - Tells us if the estimate is reliably different from zero

    **p-value**:
    - Probability of seeing this result by random chance if true value = 0
    - p < 0.05 is typically considered "statistically significant"
    - Lower p-values = stronger evidence against the null hypothesis
    """)

with st.expander("🎯 Why Does This Matter for Investing?"):
    st.markdown("""
    **For Diversified Investors**:
    - Only **systematic risk (β)** matters because firm-specific risk diversifies away
    - Stocks with β > 1 are "aggressive" and amplify market movements
    - Stocks with β < 1 are "defensive" and dampen market movements

    **For Portfolio Construction**:
    - High-beta stocks increase portfolio volatility in bull AND bear markets
    - Low-beta stocks provide stability but may underperform in rallies
    - Combining stocks with different betas helps manage overall portfolio risk

    **Alpha Interpretation**:
    - Positive α suggests the stock outperforms its risk-adjusted benchmark
    - But: α must be statistically significant to be reliable
    - Many estimated alphas are not statistically different from zero
    """)
