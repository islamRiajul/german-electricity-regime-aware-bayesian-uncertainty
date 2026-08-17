import os
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from setup_imports import DATA_DIR
from german_epf_research import (
    BSSM, DDNN, VIDDNN, ConformalWrapper, EvDNN,
    compute_all_metrics, inv_signed_log, signed_log,
)
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
compute_metrics = compute_all_metrics

def predict_log_space_interval(model, X_input, scaler_y, confidence=0.90):
    mu_s, sig_s = model.predict(X_input)
    mu_z = scaler_y.inverse_transform(mu_s.reshape(-1, 1)).flatten()
    sig_z = np.abs(sig_s) * scaler_y.scale_[0]

    z_quant = stats.norm.ppf((1 + confidence) / 2)
    z_lower, z_upper = mu_z - z_quant * sig_z, mu_z + z_quant * sig_z

    mu_eur = inv_signed_log(mu_z)
    lower_eur, upper_eur = inv_signed_log(z_lower), inv_signed_log(z_upper)
    sig_eur = (upper_eur - lower_eur) / (2 * z_quant)
    return mu_eur, np.clip(sig_eur, 1e-3, 500.0), mu_z


def tune_hyperparameters(X_tr, y_tr, X_val, y_val, y_val_raw, scaler_y, n_trials=20):
    print("\n  Running Optuna — tuning each neural model SEPARATELY...")

    def make_objective(model_name):
        def objective(trial):
            lr         = trial.suggest_float("lr", 1e-4, 2e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
            epochs     = trial.suggest_int("epochs", 100, 300, step=50)

            if model_name == "DDNN":
                model = DDNN(input_dim=X_tr.shape[1], hidden=[128, 64, 32],
                             lr=lr, epochs=epochs, batch_size=batch_size)
            elif model_name == "EvDNN":
                lam = trial.suggest_float("lam", 0.001, 0.05, log=True)
                model = EvDNN(input_dim=X_tr.shape[1], hidden=[128, 64, 32],
                              lr=lr, epochs=epochs, lam=lam, batch_size=batch_size)
            else:
                prior_sig = trial.suggest_float("prior_sigma", 0.5, 2.0)
                model = VIDDNN(input_dim=X_tr.shape[1], hidden=[128, 64],
                               lr=lr, epochs=epochs, n_samples=15,
                               prior_sigma=prior_sig, batch_size=batch_size)

            model.fit(X_tr, y_tr, verbose=False)
            mu_s, sig_s = model.predict(X_val)
            mu_z = scaler_y.inverse_transform(mu_s.reshape(-1, 1)).flatten()
            mu_eur = inv_signed_log(mu_z)
            sig_eur = np.abs(sig_s) * scaler_y.scale_[0] * np.exp(np.abs(mu_z))
            return compute_metrics(y_val_raw, mu_eur, sig_eur)["CRPS"]
        return objective

    best_by_model = {}
    for model_name in ["DDNN", "EvDNN", "VI-DDNN"]:
        study = optuna.create_study(direction="minimize")
        study.optimize(make_objective(model_name), n_trials=n_trials)
        best_by_model[model_name] = study.best_params
        print(f"    ✓ {model_name:<8} best CRPS={study.best_value:.4f}  params={study.best_params}")

    print(f"\n  ✓ Tuned all 3 neural models separately ({n_trials} trials each).")
    return best_by_model


def train_and_evaluate_pipeline(feature_set, train, val, test, best_by_model, exp_name):
    print(f"\n  --- Training Models on {exp_name} ({len(feature_set)} Features) ---")
    
    y_tr_model = signed_log(train["price"].values)
    y_val_model = signed_log(val["price"].values)
    y_te = test["price"].values

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_tr = scaler_X.fit_transform(train[feature_set].values)
    X_val = scaler_X.transform(val[feature_set].values)
    X_te = scaler_X.transform(test[feature_set].values)
    y_tr_s = scaler_y.fit_transform(y_tr_model.reshape(-1, 1)).flatten()
    y_val_s = scaler_y.transform(y_val_model.reshape(-1, 1)).flatten()

    results = {}
    def P(model):
        return best_by_model.get(model, {}) if isinstance(best_by_model, dict) else {}
    dd_p, ev_p, vi_p = P("DDNN"), P("EvDNN"), P("VI-DDNN")

    ddnn = DDNN(len(feature_set), hidden=[128, 64, 32], lr=dd_p.get("lr", 0.005), epochs=dd_p.get("epochs", 300), batch_size=dd_p.get("batch_size", 128))
    ddnn.fit(X_tr, y_tr_s, X_val, y_val_s)
    mu_eur, sig_eur, _ = predict_log_space_interval(ddnn, X_te, scaler_y)
    results["DDNN"] = compute_metrics(y_te, mu_eur, sig_eur, label=f"DDNN ({exp_name})")

    ev = EvDNN(len(feature_set), hidden=[128, 64, 32], lr=ev_p.get("lr", 0.005), epochs=ev_p.get("epochs", 300), lam=ev_p.get("lam", 0.02), batch_size=ev_p.get("batch_size", 128))
    ev.fit(X_tr, y_tr_s, X_val, y_val_s)
    mu_eur, sig_eur, _ = predict_log_space_interval(ev, X_te, scaler_y)
    results["EvDNN"] = compute_metrics(y_te, mu_eur, sig_eur, label=f"EvDNN ({exp_name})")

    vi = VIDDNN(len(feature_set), hidden=[128, 64], lr=vi_p.get("lr", 0.002), epochs=vi_p.get("epochs", 300), n_samples=20, prior_sigma=vi_p.get("prior_sigma", 1.0), batch_size=vi_p.get("batch_size", 128))
    vi.fit(X_tr, y_tr_s, X_val, y_val_s)
    mu_eur, sig_eur, mu_z = predict_log_space_interval(vi, X_te, scaler_y)
    results["VI-DDNN"] = compute_metrics(y_te, mu_eur, sig_eur, label=f"VI-DDNN ({exp_name})")

    mu_val_s, _ = vi.predict(X_val)
    mu_val_z = scaler_y.inverse_transform(mu_val_s.reshape(-1, 1)).flatten()
    mu_val_eur = inv_signed_log(mu_val_z)
    cp = ConformalWrapper(coverage=0.90)
    cp.fit(val["price"].values, mu_val_eur, val["hour"].values)
    mu_cp, sig_cp = cp.predict(mu_eur, test["hour"].values)
    results["VI+CP"] = compute_metrics(y_te, mu_cp, sig_cp, label=f"VI+CP ({exp_name})")

    bssm = BSSM(Q_init=50, R_init=300, em_iters=15)
    bssm.fit(y_tr_model)
    mu_z_bssm, sig_z_bssm = bssm.predict_test(y_tr_model, signed_log(test["price"].values))
    z90 = stats.norm.ppf(0.95)
    lower_bssm = inv_signed_log(mu_z_bssm - z90 * sig_z_bssm)
    upper_bssm = inv_signed_log(mu_z_bssm + z90 * sig_z_bssm)
    mu_bssm = inv_signed_log(mu_z_bssm)
    sig_bssm = (upper_bssm - lower_bssm) / (2 * z90)
    results["BSSM"] = compute_metrics(y_te, mu_bssm, sig_bssm, label=f"BSSM ({exp_name})")

    return results


def main():
    print("\n" + "█" * 65)
    print("  PART 8 — MODEL TRAINING & OPTUNA TUNING COMPARISON")
    print("█" * 65)

    df = pd.read_pickle(os.path.join(DATA_DIR, "data_part2.pkl"))
    with open(os.path.join(DATA_DIR, "features_part2.json")) as f:
        all_features = json.load(f)
    with open(os.path.join(DATA_DIR, "selected_part4.json")) as f:
        selected_features = json.load(f)

    train = df[df["datetime"] <= "2024-04-30 23:00"].copy()
    val = df[(df["datetime"] >= "2024-05-01") & (df["datetime"] <= "2025-04-30 23:00")].copy()
    test = df[df["datetime"] >= "2025-05-01"].copy()

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_tr_sel = scaler_X.fit_transform(train[selected_features].values)
    y_tr_s = scaler_y.fit_transform(signed_log(train["price"].values).reshape(-1, 1)).flatten()
    X_val_sel = scaler_X.transform(val[selected_features].values)
    y_val_s = scaler_y.transform(signed_log(val["price"].values).reshape(-1, 1)).flatten()

    best_by_model = tune_hyperparameters(
        X_tr_sel, y_tr_s, X_val_sel, y_val_s, val["price"].values, scaler_y, n_trials=20
    )

    sel_name = f"Selected ({len(selected_features)})"
    all_name = f"All Candidate ({len(all_features)})"

    results_selected = train_and_evaluate_pipeline(selected_features, train, val, test, best_by_model, sel_name)
    results_all = train_and_evaluate_pipeline(all_features, train, val, test, best_by_model, all_name)

    print("\n" + "=" * 80)
    print("  FEATURE SET COMPARISON TABLE (Test Set Results)")
    print("=" * 80)
    print(f"  {'Model / Feature Set':<28}{'MAE':>8}{'RMSE':>8}{'CRPS':>8}{'PICP_90':>9}{'MPIW_90':>9}{'MAACE':>8}")
    print("  " + "-" * 78)

    for name in results_selected.keys():
        r_sel = results_selected[name]
        r_all = results_all[name]
        print(f"  {name + ' (' + sel_name + ')':<28}{r_sel['MAE']:>8.2f}{r_sel['RMSE']:>8.2f}{r_sel['CRPS']:>8.2f}{r_sel['PICP_90']:>9.1%}{r_sel['MPIW_90']:>9.2f}{r_sel['MAACE']:>7.2f}%")
        print(f"  {name + ' (' + all_name + ')':<28}{r_all['MAE']:>8.2f}{r_all['RMSE']:>8.2f}{r_all['CRPS']:>8.2f}{r_all['PICP_90']:>9.1%}{r_all['MPIW_90']:>9.2f}{r_all['MAACE']:>7.2f}%")
        print("  " + "-" * 78)

    np.save(
        os.path.join(DATA_DIR, "results_part8.npy"),
        {
            "results_selected": results_selected,
            "results_all": results_all,
            "y_test": test["price"].values,
            "test_dates": test["datetime"].values,
        },
        allow_pickle=True,
    )
    print(f"\n  ✓ Saved both feature set results to {os.path.join(DATA_DIR, 'results_part8.npy')} — Ready for Part 9!")

    with open(os.path.join(DATA_DIR, "results_part8.json"), "w") as f:
        json.dump(best_by_model, f)
    print("  ✓ Saved per-model tuned hyperparameters to results_part8.json")


if __name__ == "__main__":
    main()