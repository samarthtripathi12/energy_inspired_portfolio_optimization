# Energy-Inspired Portfolio Optimization via Gradient Descent

Simulates portfolio optimization using physics-inspired energy minimization, gradient descent, and finance concepts. Combines interdisciplinary STEM, CS, and finance thinking.

---

## Abstract

This project implements a portfolio optimization framework inspired by physics energy minimization. It progressively explores risk-return trade-offs under:

1. **Simulated asset returns and covariances**  
2. **Real financial data integration**  
3. **Advanced optimization techniques (Gradient Descent, Adam)**  
4. **Efficient Frontier visualization**  
5. **Energy landscape analysis**  

The simulations combine numerical optimization, visualization, and financial theory to demonstrate practical portfolio allocation and interdisciplinary reasoning.

---

## Why This Project

- Demonstrates an interdisciplinary approach: physics → CS → finance.  
- Shows hands-on implementation of optimization algorithms.  
- Highlights advanced numerical methods (gradient descent, Adam optimizer).  
- Verifies optimization outcomes through visualization and metrics.  
- Combines static plots, animations, and energy-based visualization for storytelling.

---

## Development Iterations

- **Phase 0:** Project setup, folders, environment.  
- **Phase 1:** Parameter definition (number of assets, expected returns, covariance, hyperparameters).  
- **Phase 2:** Initial gradient descent implementation on simulated data.  
- **Phase 3:** Verification with risk-return metrics and plots.  
- **Phase 4:** Energy landscape visualization.  
- **Phase 5:** Real financial data integration.  
- **Phase 6:** Advanced optimizers (Adam vs Gradient Descent comparison).  
- **Phase 7:** Efficient Frontier & risk-return visualization.  

---

## Verification

- Loss convergence verified across iterations  
- Portfolio risk and returns tracked  
- Optimized weights validated  
- Energy terms minimized  
- Efficient frontier matches financial theory

---

## Requirements

- Python 3.11+  
- NumPy  
- Pandas  
- Matplotlib  
- SciPy  
- yfinance (Phase 5)  
- Plotly / Plotly Express (optional for interactive dashboard)  

---

## Phase 0: Setup

**Objective:** Prepare project structure and environment.

**Implementation:**
- Create project folders: `scripts/`, `data/`, `plots/`, `gifs/`  
- Create virtual environment: `project8_env`  
- Activate environment, install dependencies

**End-state / Outputs:**
- `scripts/` folder for all Python scripts  
- `data/` folder for storing CSVs  
- `plots/` folder for static plots  
- `gifs/` folder for animated outputs  

**What This Proves:**  
- Organized project structure  
- Environment ready for reproducible computation  

---

## Phase 1: Parameter Setup

**Scientific Question:**  
“What are the initial portfolio parameters and constraints?”

**Implementation:**
- Number of assets: `N = 5`  
- Expected returns: `μ = [0.05, 0.08, 0.12, 0.07, 0.09]`  
- Covariance matrix Σ (risk between asset pairs)  
- Hyperparameters: λ = 0.5 (risk-return tradeoff), γ = 0.1 (energy penalty)  
- Initial portfolio weights: `w = [0.2, 0.2, 0.2, 0.2, 0.2]`  

**End-state / Outputs:**  
- Script: `scripts/phase1_setup.py`  
- Inputs saved in Python variables  
- Ready for optimization  

**What This Proves:**  
- Clear problem definition and initial setup  
- Baseline for gradient descent optimization  

---

## Phase 2: Gradient Descent on Simulated Data

**Scientific Question:**  
“How does gradient descent optimize a portfolio using simulated returns and risk?”

**Implementation:**
- Gradient descent loop for loss minimization: `Loss = -w.T @ μ + λ * w.T @ Σ @ w + γ * ||w||^2`  
- Track: loss vs iterations, portfolio risk, return  
- Static plots: loss convergence, weight evolution

**Static Plot:**  
![Phase 2: Loss Convergence](plots/phase2_loss_convergence.png)  

**End-state / Outputs:**  
- Script: `scripts/phase2_gradient_descent.py`  
- Plots: `plots/phase2_loss_convergence.png`  
- Outputs: optimized weights

**What This Proves:**  
- Gradient descent can optimize portfolio weights  
- Energy term effectively discourages extreme allocations  

---

## Phase 3: Verification & Metrics

**Scientific Question:**  
“Are the results reliable and interpretable?”

**Implementation:**
- Verify portfolio risk and return  
- Track convergence  
- Compare final weights with expected outcomes  

**Static Plots:**  
- Portfolio weights bar chart  
- Risk vs return scatter

**End-state / Outputs:**  
- Script: `scripts/phase3_verification.py`  
- Plots: `plots/phase3_weights_bar.png`, `plots/phase3_risk_return.png`  

**What This Proves:**  
- Correct implementation of loss function and optimization  
- Verification ensures reproducibility  

---

## Phase 4: Energy Landscape Visualization

**Scientific Question:**  
“What does the optimization landscape look like?”

**Implementation:**
- 3D surface plot: Loss vs weights combinations  
- Visualize gradient descent path over energy landscape  
- Highlight minima and convergence direction  

**Static Plot:**  
![Phase 4: Energy Landscape](plots/phase4_energy_landscape.png)  

**End-state / Outputs:**  
- Script: `scripts/phase4_energy_landscape.py`  
- Plot: `plots/phase4_energy_landscape.png`  

**What This Proves:**  
- Gradient descent path follows energy minimization principles  
- Interdisciplinary link: physics-inspired energy to finance  

---

## Phase 5: Real Financial Data Integration

**Scientific Question:**  
“Can real stock data improve practical portfolio optimization?”

**Implementation:**
- Use `yfinance` to fetch historical prices for: `AAPL, MSFT, GOOGL, AMZN, TSLA`  
- Compute expected returns and covariance from real data  
- Run gradient descent on real data  
- Track optimized weights and risk-return  

**Static Plot:**  
![Phase 5: Real Data Weights](plots/phase5_real_weights.png)  

**End-state / Outputs:**  
- Script: `scripts/phase5_real_data.py`  
- Plots: `plots/phase5_real_weights.png`  
- Data CSVs: `data/real_stock_prices.csv`  

**What This Proves:**  
- Framework works with actual market data  
- Outputs practically relevant optimized portfolios  

---

## Phase 6: Advanced Optimizers (Gradient Descent vs Adam)

**Scientific Question:**  
“Do advanced optimizers converge faster and better than standard gradient descent?”

**Implementation:**
- Implement Adam optimizer alongside gradient descent  
- Compare convergence speed, loss, and final weights  
- Plot: loss vs iterations for both optimizers  

**Static Plot:**  
![Phase 6: Optimizer Comparison](plots/phase6_optimizer_comparison.png)  

**End-state / Outputs:**  
- Script: `scripts/phase6_advanced_optimizers.py`  
- Plots: `plots/phase6_optimizer_comparison.png`  

**What This Proves:**  
- Understanding of CS numerical methods  
- Adam optimizer often converges faster and avoids local minima  

---

## Phase 7: Efficient Frontier & Risk-Return Visualization

**Scientific Question:**  
“How does the portfolio perform across varying risk-return trade-offs?”

**Implementation:**
- Compute optimized portfolios for multiple λ values  
- Plot Efficient Frontier: Risk (x-axis) vs Return (y-axis)  
- Optionally animate how weights move along the frontier  

**Static Plot:**  
![Phase 7: Efficient Frontier](plots/phase7_efficient_frontier.png)  

**End-state / Outputs:**  
- Script: `scripts/phase7_efficient_frontier.py`  
- Plot: `plots/phase7_efficient_frontier.png`  

**What This Proves:**  
- Demonstrates financial maturity  
- Shows optimal portfolio selection under varying risk tolerance  

---

## Phase 8 (Optional): Interactive Dashboard

**Scientific Question:**  
“Can users interactively explore portfolio optimization?”

**Implementation:**
- Streamlit / Plotly Dash interactive sliders for λ & γ  
- Live update of optimized weights, loss, energy, and efficient frontier  
- Optional for reviewer exploration  

**End-state / Outputs:**  
- Script: `scripts/phase8_dashboard.py`  
- Runs in browser locally  

**What This Proves:**  
- Integrates CS, STEM, and finance for a fully interactive experience  

---

## Phase 9: Documentation & Storytelling

**Objective:**  
- Compile all phase-wise results  
- Embed plots, GIFs, tables  
- Explain interdisciplinary connections clearly  

**End-state / Outputs:**  
- README.md (this file)  
- Organized `plots/`, `scripts/`, `data/`, and `gifs/` folders  

**What This Proves:**  
- Strong research communication  
- Portfolio-ready, MIT-level project presentation  

---

## Conclusion

This project demonstrates **portfolio optimization inspired by physics**, progressing from:

1. Simulated asset data  
2. Gradient descent optimization  
3. Advanced optimizers (Adam)  
4. Real financial data application  
5. Efficient Frontier visualization  
6. Energy landscape understanding  

- All results are rigorously validated through static plots, animations, and quantitative metrics, demonstrating both the stability and accuracy of the optimization algorithms.  
- The project integrates **computational physics concepts, advanced optimization techniques, and real-world financial data**, offering a unique interdisciplinary perspective.  
- Phase-wise documentation, combined with energy-inspired visualizations and efficient frontier analysis, ensures full reproducibility while presenting a compelling, research-grade narrative suitable for academic or professional review.
