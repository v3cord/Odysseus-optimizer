
import os, numpy as np, pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

OUTDIR = "advanced_outputs"
os.makedirs(OUTDIR, exist_ok=True)
np.random.seed(7)

def make_synthetic_data(days=92, zones=5, start_date=datetime(2025, 8, 1)):
    hours = days * 24
    idx = pd.date_range(start_date, periods=hours, freq="H")
    data = []
    for z in range(zones):
        base_demand = np.random.uniform(200, 400)
        demand_amp = np.random.uniform(50, 120)
        wind_cap = np.random.uniform(600, 900)
        solar_cap = np.random.uniform(400, 700)

        hour = np.arange(hours) % 24
        dow = (np.arange(hours) // 24) % 7

        temp = 24 + 6*np.sin(2*np.pi*(hour-13)/24) + np.random.normal(0, 1.2, hours)
        solar_irr = np.clip(np.sin(2*np.pi*(hour-6)/24), 0, None)
        solar_irr = solar_irr ** 1.2 + np.random.normal(0, 0.04, hours)
        solar_irr = np.clip(solar_irr, 0, 1)
        t = np.arange(hours)
        wind = 0.5 + 0.3*np.sin(2*np.pi*t/168 + z) + 0.2*np.random.rand(hours)
        wind = np.clip(wind, 0, 1)

        base_price = 5.0 + 0.5*np.sin(2*np.pi*(hour-10)/24) + 0.3*np.sin(2*np.pi*t/168)
        price = base_price + np.random.normal(0, 0.2, hours)
        price = np.clip(price, 3.5, 7.5)
        price_mwh = price * 1000

        demand = base_demand \
                 + demand_amp*np.sin(2*np.pi*(hour-20)/24) \
                 + 0.6*(temp-24) \
                 + 20*(dow>=5) \
                 + np.random.normal(0, 12, hours)
        demand = np.clip(demand, 50, None)

        wind_gen = wind_cap * (wind**1.5) * (0.9 + 0.1*np.random.rand(hours))
        solar_gen = solar_cap * solar_irr * (0.95 + 0.05*np.random.rand(hours))
        gen = np.clip(wind_gen + solar_gen, 0, None)

        df = pd.DataFrame({
            "zone": f"Z{z+1}",
            "ts": idx,
            "hour": hour,
            "dow": dow,
            "temp": temp,
            "solar_irr": solar_irr,
            "wind": wind,
            "price_inr_per_mwh": price_mwh,
            "demand_mw": demand,
            "gen_mw": gen,
        })
        data.append(df)
    return pd.concat(data, ignore_index=True)

def train_quantile_models(X, y, base_params):
    m10 = GradientBoostingRegressor(loss="quantile", alpha=0.1, **base_params).fit(X, y)
    m50 = GradientBoostingRegressor(loss="squared_error", **base_params).fit(X, y)
    m90 = GradientBoostingRegressor(loss="quantile", alpha=0.9, **base_params).fit(X, y)
    return m10, m50, m90

def main():
    df = make_synthetic_data()

    # Horizon
    cutoff = df["ts"].min() + timedelta(days=90)
    future_idx = pd.date_range(cutoff, periods=48, freq="H")

    # ---------------- Generation forecast (per zone) ----------------
    features_gen = ["hour","dow","temp","solar_irr","wind"]
    gb_params = dict(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=7)

    gen_forecasts = []
    zone_models_gen = {}
    for z, zdf in df.groupby("zone"):
        train = zdf[zdf["ts"] < cutoff]
        future = zdf[zdf["ts"].isin(future_idx)].copy()
        X = train[features_gen].values
        y = train["gen_mw"].values
        m10, m50, m90 = train_quantile_models(X, y, gb_params)
        zone_models_gen[z] = (m10, m50, m90)

        Xf = future[features_gen].values
        future["gen_p10"] = m10.predict(Xf)
        future["gen_p50"] = m50.predict(Xf)
        future["gen_p90"] = m90.predict(Xf)
        gen_forecasts.append(future[["zone","ts","gen_p10","gen_p50","gen_p90"]])
    gen_forecast_df = pd.concat(gen_forecasts, ignore_index=True)

    # ---------------- Demand forecast (per zone + total quantile) ---------------
    features_dem = ["hour","dow","temp"]
    dem_forecasts = []
    zone_models_dem = {}
    for z, zdf in df.groupby("zone"):
        train = zdf[zdf["ts"] < cutoff]
        future = zdf[zdf["ts"].isin(future_idx)].copy()
        X = train[features_dem].values
        y = train["demand_mw"].values
        m = GradientBoostingRegressor(loss="squared_error", n_estimators=250, max_depth=3, learning_rate=0.05, random_state=9).fit(X, y)
        zone_models_dem[z] = m
        future["demand_hat"] = m.predict(future[features_dem].values)
        dem_forecasts.append(future[["zone","ts","demand_hat"]])
    dem_forecast_df = pd.concat(dem_forecasts, ignore_index=True)

    # Total demand quantiles
    hist_agg = df[df["ts"] < cutoff].groupby("ts")["demand_mw"].sum().reset_index(name="demand_total")
    hist_feat = df[df["ts"] < cutoff][["ts","hour","dow","temp"]].drop_duplicates().sort_values("ts")
    hist = hist_agg.merge(hist_feat, on="ts", how="left")
    q10, q50, q90 = train_quantile_models(hist[["hour","dow","temp"]].values, hist["demand_total"].values, gb_params)

    future_feats = df[df["ts"].isin(future_idx)][["ts","hour","dow","temp"]].drop_duplicates().sort_values("ts")
    future_feats["dem_p10"] = q10.predict(future_feats[["hour","dow","temp"]].values)
    future_feats["dem_p50"] = q50.predict(future_feats[["hour","dow","temp"]].values)
    future_feats["dem_p90"] = q90.predict(future_feats[["hour","dow","temp"]].values)

    # ---------------- Market price forecast ------------------------------------
    hist_price = df[df["ts"] < cutoff].groupby("ts").agg({
        "gen_mw":"sum","demand_mw":"sum","price_inr_per_mwh":"mean"
    }).reset_index()
    hist_price["hour"] = hist_price["ts"].dt.hour
    hist_price["dow"] = hist_price["ts"].dt.dayofweek
    p10, p50, p90 = train_quantile_models(hist_price[["gen_mw","demand_mw","hour","dow"]].values,
                                          hist_price["price_inr_per_mwh"].values, gb_params)

    gen_total_future = gen_forecast_df.groupby("ts")["gen_p50"].sum().reset_index(name="gen_total")
    dem_total_future = dem_forecast_df.groupby("ts")["demand_hat"].sum().reset_index(name="demand_total")
    f = pd.DataFrame({"ts": future_idx})
    f["hour"] = f["ts"].dt.hour
    f["dow"] = f["ts"].dt.dayofweek
    f = f.merge(gen_total_future, on="ts").merge(dem_total_future, on="ts")
    Xf = f[["gen_total","demand_total","hour","dow"]].values
    f["price_p10"] = p10.predict(Xf)
    f["price_p50"] = p50.predict(Xf)
    f["price_p90"] = p90.predict(Xf)

    # ---------------- Save charts (presentation-ready) -------------------------
    flag_zone = "Z1"
    z_hist = df[(df["zone"]==flag_zone) & (df["ts"] >= cutoff - timedelta(hours=24)) & (df["ts"] < cutoff)]
    z_fore = gen_forecast_df[gen_forecast_df["zone"]==flag_zone].copy()

    plt.figure(figsize=(14,6))
    plt.plot(z_hist["ts"], z_hist["gen_mw"], label="Actual Gen (last 24h)")
    plt.plot(z_fore["ts"], z_fore["gen_p50"], label="Forecast Gen P50 (next 48h)")
    plt.fill_between(z_fore["ts"].values, z_fore["gen_p10"].values, z_fore["gen_p90"].values, alpha=0.2, label="P10–P90 band")
    plt.title(f"{flag_zone} Generation Forecast: Last 24h Actual + Next 48h Forecast")
    plt.xlabel("Time"); plt.ylabel("MW"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "gen_forecast_zone1.png")); plt.close()

    dem_piv = dem_forecast_df.pivot(index="ts", columns="zone", values="demand_hat").reindex(future_idx)
    plt.figure(figsize=(14,6))
    plt.stackplot(future_idx, [dem_piv[col].values for col in dem_piv.columns], labels=list(dem_piv.columns))
    plt.title("Predicted Demand by Zone (Next 48h)")
    plt.xlabel("Time"); plt.ylabel("MW"); plt.legend(loc="upper left", ncol=3, fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "demand_forecast_stacked.png")); plt.close()

    plt.figure(figsize=(14,6))
    plt.plot(future_feats["ts"], future_feats["dem_p50"], label="Demand P50 (Total)")
    plt.fill_between(future_feats["ts"].values, future_feats["dem_p10"].values, future_feats["dem_p90"].values, alpha=0.2, label="P10–P90 band")
    plt.title("Total Demand Forecast with Uncertainty (Next 48h)")
    plt.xlabel("Time"); plt.ylabel("MW"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "demand_total_band.png")); plt.close()

    plt.figure(figsize=(14,6))
    plt.plot(f["ts"], f["price_p50"], label="Price P50 (₹/MWh)")
    plt.fill_between(f["ts"].values, f["price_p10"].values, f["price_p90"].values, alpha=0.2, label="P10–P90 band")
    thr = np.percentile(f["price_p50"], 90)
    hp = f[f["price_p50"] >= thr]
    plt.scatter(hp["ts"], hp["price_p50"], label="High-price hours (top 10%)")
    plt.title("Market Price Forecast & High-Price Windows (Next 48h)")
    plt.xlabel("Time"); plt.ylabel("₹/MWh"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "price_forecast_band.png")); plt.close()

    # Feature importance for explainability
    _, m50, _ = zone_models_gen[flag_zone]
    plt.figure(figsize=(10,6))
    plt.bar(features_gen, m50.feature_importances_)
    plt.title(f"{flag_zone} Generation Model – Feature Importance")
    plt.xlabel("Feature"); plt.ylabel("Importance"); plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "gen_feature_importance.png")); plt.close()

    # Save CSVs
    gen_forecast_df.to_csv(os.path.join(OUTDIR, "gen_forecast_48h.csv"), index=False)
    dem_forecast_df.to_csv(os.path.join(OUTDIR, "demand_forecast_48h.csv"), index=False)
    f.to_csv(os.path.join(OUTDIR, "price_forecast_48h.csv"), index=False)

if __name__ == "__main__":
    main()
