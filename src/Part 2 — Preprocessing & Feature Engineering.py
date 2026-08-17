import os
import json
import numpy as np
import pandas as pd
from setup_imports import DATA_DIR

def engineer_features(df):
    print("Engineering advanced features...")
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].ffill().bfill().fillna(0)

    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f"lag_{lag}"] = df["price"].shift(lag)

    df["roll_mean_24"] = df["price"].shift(1).rolling(24).mean()
    df["roll_std_24"] = df["price"].shift(1).rolling(24).std()
    df["roll_mean_168"] = df["price"].shift(1).rolling(168).mean()
    df["price_change"] = df["price"].shift(1) - df["price"].shift(25)

    df["ema_6h"] = df["price"].shift(1).ewm(span=6).mean()
    df["ema_24h"] = df["price"].shift(1).ewm(span=24).mean()
    df["ema_72h"] = df["price"].shift(1).ewm(span=72).mean()

    df["total_renewable"] = (
        df["wind_offshore"] + df["wind_onshore"] + df["solar"]
    )
    df["ren_share"] = df["total_renewable"] / (df["load_forecast"] + 1e-6)
    df["demand_ren_ratio"] = df["load_forecast"] / (
        df["total_renewable"] + 1e-6
    )

    df["solar_ramp_1h"] = df["solar"] - df["solar"].shift(1)
    df["solar_ramp_3h"] = df["solar"] - df["solar"].shift(3)
    df["wind_ramp_1h"] = df["total_renewable"] - df["total_renewable"].shift(1)
    df["load_ramp_1h"] = df["load_forecast"] - df["load_forecast"].shift(1)

    df["hour"] = df["datetime"].dt.hour
    df["dow"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df = df.dropna().reset_index(drop=True)

    feature_names = [
        "lag_1", "lag_2", "lag_3", "lag_6", "lag_12", "lag_24", "lag_48", "lag_168",
        "roll_mean_24", "roll_std_24", "roll_mean_168", "price_change",
        "ema_6h", "ema_24h", "ema_72h",
        "load_forecast", "residual_load", "total_gen", "pv_wind", "ren_share", "demand_ren_ratio",
        "solar_ramp_1h", "solar_ramp_3h", "wind_ramp_1h", "load_ramp_1h",
        "wind_offshore", "wind_onshore", "solar", "other_gen",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    ]
    feature_names = [f for f in feature_names if f in df.columns]

    print(f"  ✓ Created {len(feature_names)} candidate features")
    print(f"  ✓ Rows after cleaning: {len(df):,}")
    return df, feature_names


def split_train_val_test(df):
    train = df[df["datetime"] <= "2024-04-30 23:00"].copy()
    val = df[
        (df["datetime"] >= "2024-05-01") & (df["datetime"] <= "2025-04-30 23:00")
    ].copy()
    test = df[df["datetime"] >= "2025-05-01"].copy()
    return train, val, test


if __name__ == "__main__":
    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part1.pkl"))
    df, features = engineer_features(df)
    train, val, test = split_train_val_test(df)

    df.to_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "features_part2.json"), "w") as f:
        json.dump(features, f)

    print("\n✓ Saved to data_part2.pkl — Ready for Part 4 feature selection!")