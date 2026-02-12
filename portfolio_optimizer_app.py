import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Optimal Portfolio Calculator", layout="wide", page_icon="📊")

# Title and description
st.title("📊 Optimal Complete Portfolio Calculator")
st.markdown("**FIN 684 - Investment Analysis and Portfolio Management | Dr. Efe Cotelioglu**")
st.markdown("---")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

st.sidebar.subheader("Asset 1: Bonds")
E_rB = st.sidebar.number_input("Expected Return E(r_B) (%)", value=5.0, min_value=0.0, max_value=50.0, step=0.1) / 100
sigma_B = st.sidebar.number_input("Standard Deviation σ_B (%)", value=8.0, min_value=0.1, max_value=100.0, step=0.1) / 100

st.sidebar.subheader("Asset 2: Stocks")
E_rS = st.sidebar.number_input("Expected Return E(r_S) (%)", value=10.0, min_value=0.0, max_value=50.0, step=0.1) / 100
sigma_S = st.sidebar.number_input("Standard Deviation σ_S (%)", value=19.0, min_value=0.1, max_value=100.0, step=0.1) / 100

st.sidebar.subheader("Correlation & Risk-Free Rate")
rho_BS = st.sidebar.slider("Correlation ρ_BS", min_value=-1.0, max_value=1.0, value=0.20, step=0.01)
r_f = st.sidebar.number_input("Risk-Free Rate r_f (%)", value=3.0, min_value=0.0, max_value=20.0, step=0.1) / 100

st.sidebar.subheader("Investor Risk Aversion")
A = st.sidebar.slider("Risk Aversion Coefficient (A)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Adjust the parameters to see how they affect the optimal portfolio.")

# Calculate covariance
cov_BS = rho_BS * sigma_B * sigma_S

# Calculate optimal risky portfolio weights
try:
    numerator = (E_rB - r_f) * sigma_S**2 - (E_rS - r_f) * sigma_B * sigma_S * rho_BS
    denominator = (E_rB - r_f) * sigma_S**2 + (E_rS - r_f) * sigma_B**2 - (E_rB - r_f + E_rS - r_f) * sigma_B * sigma_S * rho_BS
    
    if abs(denominator) < 1e-10:
        st.error("❌ Invalid inputs: Cannot calculate optimal portfolio with these parameters.")
        st.stop()
    
    w_B = numerator / denominator
    w_S = 1 - w_B
    
    # Calculate E(r_O) and sigma_O
    E_rO = w_B * E_rB + w_S * E_rS
    sigma_O_squared = w_B**2 * sigma_B**2 + w_S**2 * sigma_S**2 + 2 * w_B * w_S * sigma_B * sigma_S * rho_BS
    
    if sigma_O_squared < 0:
        st.error("❌ Invalid inputs: Negative variance calculated.")
        st.stop()
    
    sigma_O = np.sqrt(sigma_O_squared)
    
    # Sharpe ratio
    sharpe_ratio = (E_rO - r_f) / sigma_O if sigma_O > 0 else 0
    
    # Calculate optimal allocation to risky portfolio
    y_star = (E_rO - r_f) / (A * sigma_O_squared) if sigma_O_squared > 0 else 0
    
    # Complete portfolio characteristics
    E_rC = r_f + y_star * (E_rO - r_f)
    sigma_C = abs(y_star) * sigma_O
    
    # Final allocations
    alloc_B = y_star * w_B
    alloc_S = y_star * w_S
    alloc_rf = 1 - y_star
    
    # Generate opportunity set
    weights_B = np.linspace(0, 1, 100)
    weights_S = 1 - weights_B
    
    E_portfolio = weights_B * E_rB + weights_S * E_rS
    sigma_portfolio = np.sqrt(
        weights_B**2 * sigma_B**2 + 
        weights_S**2 * sigma_S**2 + 
        2 * weights_B * weights_S * sigma_B * sigma_S * rho_BS
    )
    
    # Find minimum variance portfolio
    min_var_idx = np.argmin(sigma_portfolio)
    
    # Generate CAL
    sigma_CAL = np.linspace(0, max(sigma_S * 1.5, sigma_C * 1.5), 100)
    E_CAL = r_f + sharpe_ratio * sigma_CAL
    
    # Generate indifference curve through C
    U_C = E_rC - 0.5 * A * sigma_C**2
    sigma_indiff = np.linspace(max(0.001, sigma_C * 0.5), sigma_C * 2, 100)
    E_indiff = U_C + 0.5 * A * sigma_indiff**2
    
    # Create two columns for layout
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📈 Optimal Portfolio Visualization")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot opportunity set
        ax.plot(sigma_portfolio * 100, E_portfolio * 100, 'b-', linewidth=2.5, label='Opportunity Set')
        
        # Plot CAL
        ax.plot(sigma_CAL * 100, E_CAL * 100, 'r-', linewidth=2.5, label='CAL', zorder=5)
        
        # Plot indifference curve
        ax.plot(sigma_indiff * 100, E_indiff * 100, 'g-', linewidth=2, label='Indifference Curve', alpha=0.7)
        
        # Mark individual assets
        ax.plot(sigma_B * 100, E_rB * 100, 'gs', markersize=14, label='Bond (B)', zorder=10)
        ax.text(sigma_B * 100 - 0.8, E_rB * 100, 'B', fontsize=11, ha='right', va='center', 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.plot(sigma_S * 100, E_rS * 100, 'bs', markersize=14, label='Stock (S)', zorder=10)
        ax.text(sigma_S * 100 + 0.8, E_rS * 100, 'S', fontsize=11, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Mark risk-free rate
        ax.plot(0, r_f * 100, 'ro', markersize=10, zorder=10)
        ax.text(-0.5, r_f * 100, f'$r_f$', fontsize=11, ha='right', va='center')
        
        # Mark optimal risky portfolio P
        ax.plot(sigma_O * 100, E_rO * 100, 'ko', markersize=12, zorder=10)
        ax.text(sigma_O * 100 + 0.5, E_rO * 100 - 0.3, 'P', fontsize=11, ha='left', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # Mark optimal complete portfolio C
        ax.plot(sigma_C * 100, E_rC * 100, 'mo', markersize=12, zorder=10)
        ax.text(sigma_C * 100 - 0.5, E_rC * 100, 'C', fontsize=11, ha='right', va='center',
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Formatting
        ax.set_xlabel('Standard Deviation (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Expected Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Determination of the Optimal Complete Portfolio', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='lower right')
        
        # Set reasonable axis limits
        max_sigma = max(sigma_S * 100, sigma_C * 100) * 1.3
        max_return = max(E_rS * 100, E_rC * 100) * 1.2
        ax.set_xlim(0, max_sigma)
        ax.set_ylim(0, max_return)
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("📊 Results")
        
        # Optimal Risky Portfolio
        st.markdown("### 🎯 Optimal Risky Portfolio (P)")
        results_P = pd.DataFrame({
            'Metric': ['Weight in Bonds', 'Weight in Stocks', 'Expected Return', 'Standard Deviation', 'Sharpe Ratio'],
            'Value': [
                f'{w_B*100:.2f}%',
                f'{w_S*100:.2f}%',
                f'{E_rO*100:.2f}%',
                f'{sigma_O*100:.2f}%',
                f'{sharpe_ratio:.4f}'
            ]
        })
        st.dataframe(results_P, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Optimal Complete Portfolio
        st.markdown(f"### 💼 Optimal Complete Portfolio (A = {A:.1f})")
        
        position_type = "Leveraged (Borrowing)" if y_star > 1 else "Conservative (Lending)" if y_star < 1 else "100% in Risky Portfolio"
        
        results_C = pd.DataFrame({
            'Metric': ['Allocation to P (y*)', 'Position Type', 'Expected Return', 'Standard Deviation'],
            'Value': [
                f'{y_star*100:.2f}%',
                position_type,
                f'{E_rC*100:.2f}%',
                f'{sigma_C*100:.2f}%'
            ]
        })
        st.dataframe(results_C, hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Final Asset Allocation
        st.markdown("### 🏦 Final Asset Allocation")
        final_alloc = pd.DataFrame({
            'Asset': ['Bonds', 'Stocks', 'Risk-Free'],
            'Allocation': [
                f'{alloc_B*100:.2f}%',
                f'{alloc_S*100:.2f}%',
                f'{alloc_rf*100:.2f}%'
            ]
        })
        st.dataframe(final_alloc, hide_index=True, use_container_width=True)
        
        # Color code the position
        if y_star > 1:
            st.warning(f"⚠️ This investor is **leveraged**, borrowing {abs(alloc_rf)*100:.2f}% to invest more in the risky portfolio.")
        elif y_star < 1:
            st.success(f"✅ This investor is **conservative**, holding {alloc_rf*100:.2f}% in risk-free assets.")
        else:
            st.info("ℹ️ This investor is **fully invested** in the risky portfolio with no leverage or risk-free holdings.")

except Exception as e:
    st.error(f"❌ Error in calculations: {str(e)}")
    st.stop()

# Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p style='color: gray; font-size: 12px;'>
        Built for FIN 684 - Investment Analysis and Portfolio Management<br>
        American University of Sharjah | Dr. Efe Cotelioglu
    </p>
</div>
""", unsafe_allow_html=True)
