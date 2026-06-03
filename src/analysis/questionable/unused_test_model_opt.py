import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Set python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.analysis.shadow_price_forecaster import build_features

def test_optimization():
    data_dir = Path(__file__).parent.parent.parent.parent / "src" / "data" / "timeseries" / "constraints"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        print("No historical parquet files found.")
        return

    dfs = []
    for f in parquet_files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            continue
    if not dfs:
        print("Could not load any data.")
        return
    
    df = pd.concat(dfs).sort_values("interval").reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]
    
    binding_counts = df[df["shadow_price"].abs() > 0.01].groupby("constraint_name").size().sort_values(ascending=False)
    test_constraints = list(binding_counts[binding_counts > 50].index[:6])
    
    # Let's also include the specific problematic ones mentioned in the implementation status if present:
    for spec in ["PSOSWEPCOTIE", "FPLWDWCLEMEN"]:
        if spec in df["constraint_name"].unique() and spec not in test_constraints:
            test_constraints.append(spec)
            
    print(f"Testing on constraints: {test_constraints}\n")
    
    # We will try adding binding frequency features and test 6 model configs
    def build_features_opt(history):
        feat = build_features(history)
        sp = history["shadow_price"]
        for w in (6, 24, 72):
            feat[f"sp_roll_bind_freq_{w}"] = (sp.abs() > 0.01).shift(1).rolling(w, min_periods=1).mean()
        return feat

    for c in test_constraints:
        c_df = df[df["constraint_name"] == c].copy().sort_values("interval").reset_index(drop=True)
        if len(c_df) < 50:
            continue
            
        n_test = max(1, int(len(c_df) * 0.2))
        train_df = c_df.iloc[:-n_test]
        test_df = c_df.iloc[-n_test:]
        y_train = train_df["shadow_price"].values.astype(float)
        y_test = test_df["shadow_price"].values.astype(float)
        
        # Original features
        feat_df_orig = build_features(c_df)
        cols_orig = [col for col in feat_df_orig.columns if col != "interval"]
        X_orig = feat_df_orig[cols_orig].values.astype(float)
        X_train_orig, X_test_orig = X_orig[:-n_test], X_orig[-n_test:]
        
        # Opt features (with bind frequency)
        feat_df_opt = build_features_opt(c_df)
        cols_opt = [col for col in feat_df_opt.columns if col != "interval"]
        X_opt = feat_df_opt[cols_opt].values.astype(float)
        X_train_opt, X_test_opt = X_opt[:-n_test], X_opt[-n_test:]
        
        print(f"=== Constraint: {c} (Total: {len(c_df)}, Train: {len(y_train)}, Test: {len(y_test)}) ===")
        
        configs = {
            "1. Orig (Raw target, MSE loss, Orig features)": (X_train_orig, X_test_orig, y_train, "squared_error", False),
            "2. Raw target, MAE loss, Orig features": (X_train_orig, X_test_orig, y_train, "absolute_error", False),
            "3. Symlog target, MSE loss, Orig features": (X_train_orig, X_test_orig, y_train, "squared_error", True),
            "4. Symlog target, MAE loss, Orig features": (X_train_orig, X_test_orig, y_train, "absolute_error", True),
            "5. Raw target, MAE loss, Opt features (bind freq)": (X_train_opt, X_test_opt, y_train, "absolute_error", False),
            "6. Symlog target, MAE loss, Opt features (bind freq)": (X_train_opt, X_test_opt, y_train, "absolute_error", True),
        }
        
        for name, (X_tr, X_te, y_tr, loss, use_symlog) in configs.items():
            if use_symlog:
                y_tr_fit = np.sign(y_tr) * np.log1p(np.abs(y_tr))
            else:
                y_tr_fit = y_tr
                
            model = HistGradientBoostingRegressor(loss=loss, max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=5, random_state=42)
            model.fit(X_tr, y_tr_fit)
            
            y_pred_fit = model.predict(X_te)
            
            if use_symlog:
                y_pred = np.sign(y_pred_fit) * (np.expm1(np.abs(y_pred_fit)))
            else:
                y_pred = y_pred_fit
                
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  {name:<55} -> MAE: {mae:8.4f} | R2: {r2:8.4f}")
        print()

if __name__ == "__main__":
    test_optimization()
