import os
import numpy as np
import pandas as pd
from setup_imports import DATA_DIR

def parse_european_number(s):
    if pd.isna(s):
        return np.nan
    return pd.to_numeric(str(s).replace(',', ''), errors='coerce')


def load_dataset(path):
    print("Loading SMARD dataset...")
    df = pd.read_csv(path, sep=';', skiprows=1, encoding='utf-8')

    rename_map = {
        'Germany/Luxembourg [€/MWh] Calculated resolutions':  'price',
        'grid load [MWh] Calculated resolutions':             'load_forecast',
        'Residual load [MWh] Calculated resolutions':         'residual_load',
        'Total [MWh] Calculated resolutions':                 'total_gen',
        'Photovoltaics and wind [MWh] Calculated resolutions':'pv_wind',
        'Wind offshore [MWh] Calculated resolutions':         'wind_offshore',
        'Wind onshore [MWh] Calculated resolutions':          'wind_onshore',
        'Photovoltaics [MWh] Calculated resolutions':         'solar',
        'Other [MWh] Calculated resolutions':                 'other_gen',
    }
    df = df.rename(columns=rename_map)

    balancing_cols = [
        'Volume (+) [MWh] Calculated resolutions',
        'Volume (-) [MWh] Calculated resolutions',
        'Price [€/MWh] Calculated resolutions',
        'Net income [€] Calculated resolutions',
    ]
    df = df.drop(columns=[c for c in balancing_cols if c in df.columns])
    print(f"  Dropped balancing columns: Volume(+), Volume(-), Price, Net income")

    numeric_cols = ['price', 'load_forecast', 'residual_load', 'total_gen',
                    'pv_wind', 'wind_offshore', 'wind_onshore', 'solar', 'other_gen']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_european_number)

    df['datetime'] = pd.to_datetime(df['Start date'],
                                    format='%b %d, %Y %I:%M %p',
                                    errors='coerce')

    df = df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
    df = df.drop(columns=['Start date', 'End date'])

    print(f"  Loaded {len(df):,} hourly rows")
    print(f"  Date range: {df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()}")
    print(f"  Price range: €{df['price'].min():.1f} to €{df['price'].max():.1f}")
    return df


if __name__ == "__main__":
    PATH = os.path.join(DATA_DIR, "Day-ahead_prices_202001010000_202605010000_Hour.csv")
    df = load_dataset(PATH)

    print("\nFirst 5 rows:")
    print(df[['datetime', 'price', 'load_forecast', 'wind_onshore', 'solar']].head())

    df.to_pickle(os.path.join(DATA_DIR, "data_part1.pkl"))
    print("\n✓ Saved to data_part1.pkl — ready for Part 2")