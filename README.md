# ⚡ Project ASTRIDE: AI-Powered Renewable Energy Dispatch Optimizer

> **AI-based Decision Framework** for Demand–Supply Forecasting, Storage Optimization & Dynamic Bidding  
> 🚀 Designed to align with **India’s IEX market** and renewable energy integration goals.

---

## 🌍 Problem Context

India’s renewable power producers face three major challenges:
1. ⚖️ **Demand–Supply mismatch** across regions  
2. 🌦️ **Weather-driven generation fluctuations** (solar & wind)  
3. 🔋 **Inefficient utilization** of energy storage and transmission assets  

Astride aims to solve this by combining **AI forecasting models** with **real-time optimization** using machine learning and linear programming.

---

## 🎯 Project Objectives

| Goal | Target | Status (Achieved) |
|------|---------|------------------|
| 🔌 Grid Reliability | +15% improvement (from 82% → ≥94%) | ✅ ~95% |
| ⚙️ Energy Loss Reduction | −20% vs baseline (≤9%) | ✅ ~6–8% |
| 💰 EBITDA Margin | ≥15% | ✅ ~20–25% |

---

## 🧩 Technical Overview

### 🛠️ Core Components

| Module | Function |
|--------|-----------|
| `data generation` | Synthesizes 5-zone hourly data for 90+ days (weather, demand, generation, prices) |
| `forecasting` | ML models (Gradient Boosting) predict next 48 hours for **generation**, **demand**, **prices** |
| `optimization` | Linear Programming (PuLP) allocates energy, optimizes storage (Battery + Hydro), and bids on IEX |
| `kpi computation` | Calculates Reliability, Loss%, EBITDA Margin |
| `visualization` | Generates clear performance charts & saves results to CSV |

---

## ⚙️ Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| 💻 Programming | Python 3.11+ |
| 📈 Forecasting | `scikit-learn` (Gradient Boosting Regressor) |
| 🔢 Optimization | `PuLP` (Linear Programming Solver) |
| 📊 Data Handling | `pandas`, `numpy` |
| 🎨 Visualization | `matplotlib` |
| 📂 Environment | `VS Code`, `venv` (Virtual Environment) |

---

## 📁 Folder Structure

```plaintext
astride-mini-starter/
|
├── astride_final_case.py # 🚀 Main Python Script (run this!)
├── final_outputs/ # 📊 Outputs (charts + CSV)
│   ├── dispatch_results.csv # Full dispatch log (hourly per zone)
│   ├── kpi_summary.png # KPI bar chart (Reliability, Loss%, EBITDA%)
│   └── energy_sold_zones.png # Energy sold by zone visualization
|
├── advanced_outputs/ # 🔎 Forecast & zone-level analysis
|
└── README.md # 📖 You're reading this file!


---

## 🧪 How It Works (Step-by-Step)

### 🔹 Step 1 — Data Generation
- Creates **synthetic data** for 5 renewable zones
- Variables:
  - `solar_irr`, `wind`, `temp`
  - `gen_mw`, `demand_mw`
  - `price_inr_per_mwh`
- 5 years of hourly data simulated, sample cut for last 48 hours forecasting.

### 🔹 Step 2 — Forecasting (Machine Learning)
- Uses **Gradient Boosting Regressor** to learn:
  - Generation = f(solar_irr, wind, temp)
  - Demand = f(hour, day, temp)
- Forecast horizon = **48 hours**
- Predicts:  
  🟢 `gen_p50`, 🔵 `demand_hat`, 🟠 `price_fc`

### 🔹 Step 3 — Optimization (AI Dispatch)
- For each hour and zone:
  - Decides how much to:
    - Serve to demand (retail)
    - Sell to market (IEX)
    - Charge/Discharge battery or hydro
  - Enforces:
    - ⚖️ Energy balance  
    - 🔋 State-of-Charge constraints  
    - 🧾 Reserve margin (5%)  
    - 💸 Market price limits (±20% of IEX)  
- Solver: **PuLP CBC**

### 🔹 Step 4 — KPI Evaluation
| KPI | Formula | Meaning |
|------|----------|---------|
| **Reliability** | (Energy Served ÷ Forecast Demand) | How much demand was met |
| **Loss%** | (Tx + Storage losses ÷ Generation) | Energy lost in process |
| **EBITDA%** | (Revenue – Opex) ÷ Revenue | Profitability indicator |

### 🔹 Step 5 — Visualization
- Bar chart: **Reliability, Loss%, EBITDA%**
- Line chart: **Energy sold per zone**
- CSV logs for reproducibility.

---

## 🧰 How to Run the Project

### 🧱 1. Create Environment
```bash
python -m venv .venv
```
### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn pulp matplotlib
```

### 3. Run the main script and advanced output script
```bash
python astride_final_case.py
python astride_forecast_advanced.py
```

Grid Reliability: 94%  (Target: 94.3%)
Energy Loss %:    2.54%  (Target: 8.8%)
EBITDA Margin:    19.16%  (Target: > 15.0%)
Saved CSV: final_outputs/dispatch_results.csv
Charts saved in: final_outputs/

---

## 🤖 Advanced Forecasting Module

The **`advanced_outputs/`** folder stores results from the *AI forecasting engine* built using **Gradient Boosting Regressors**.  
This module predicts generation, demand, and price trends for the next 48 hours — providing inputs for the optimization layer.

### 🔍 What It Does
| Forecast Type | Learned From | Predicts | Purpose |
|----------------|--------------|-----------|----------|
| ☀️ **Generation Forecast** | Solar irradiance, wind speed, temperature | `gen_p50` | Estimate renewable generation potential |
| ⚡ **Demand Forecast** | Hour, day of week, temperature | `demand_hat` | Predict future regional demand |
| 💸 **Market Price Forecast** | Historical IEX prices, demand & generation | `price_fc` | Estimate price trends for bidding strategy |

---

### 📂 Files in `advanced_outputs/`

| 🗂️ File | 📖 Description |
|----------|----------------|
| `gen_forecast_48h.csv` | 48-hour renewable generation forecast per zone |
| `demand_forecast_48h.csv` | 48-hour demand forecast per zone |
| `price_forecast_48h.csv` | 48-hour dynamic market price predictions |
| `gen_total_band.png` | Confidence band (P10–P90) for aggregated generation |
| `demand_total_band.png` | Confidence band for aggregated demand |
| `gen_feature_importance.png` | ML feature importance (weather vs output impact) |

---

### 🧠 Model Details

- Algorithm: **Gradient Boosting Regressor (GBR)**  
- Framework: `scikit-learn`
- Input features:
  - `hour`, `dow`, `temp`, `solar_irr`, `wind`
- Target variables:
  - `gen_mw`, `demand_mw`, `price_inr_per_mwh`
- Forecast Horizon: **48 hours**  
- Output metrics visualized in PNGs and exported to CSVs for analysis.

---

### 📈 Visualization Example

Each forecast plot shows:
- 🟢 **P50 (mean prediction)** → Most probable trend  
- 🟡 **P10 & P90 bands** → Lower & upper uncertainty limits  
- 🔵 **Actual generation/demand** overlay for validation  

These visualizations help operators **anticipate shortfalls or surpluses**, improving bidding and storage decisions.

---

### 🧩 Integration with Optimization
- Forecasted values (`gen_p50`, `demand_hat`, `price_fc`) feed directly into the **linear programming optimizer** in `astride_final_case.py`.
- This creates a **closed-loop decision system** that balances:
  - 📊 *Forecasting accuracy*  
  - ⚡ *Operational constraints*  
  - 💰 *Profit maximization*

---

> 💡 **Pro Tip:** You can re-train or extend the forecasting window (e.g. 7 days) by modifying the variable `HOURS = 48` in the main script.

---

