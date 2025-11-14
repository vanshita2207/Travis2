import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# CONFIG
# ---------------------------
CSV1 = "Traffic.csv"
CSV2 = "TrafficTwoMonth.csv"
YEAR_PLACEHOLDER = 2023
FREQ_MINUTES = 15
LOOKAHEAD_HOURS = 48
SAMPLE_PER_HOUR = 60 // FREQ_MINUTES
LOOKAHEAD_STEPS = LOOKAHEAD_HOURS * SAMPLE_PER_HOUR  # 192
TEST_RATIO = 0.20
MODEL_FILE = "model_xgb.joblib"

# ---------------------------
# Load / merge
# ---------------------------
def load_and_merge(csv_paths):
    frames = []
    for p in csv_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"CSV not found: {p}")
        df = pd.read_csv(p)
        frames.append(df)
        print(f"Loaded {p} shape={df.shape}")
    df_all = pd.concat(frames, ignore_index=True)
    print("Merged dataframe shape:", df_all.shape)
    return df_all

# ---------------------------
# Robust timestamp rebuild with gap detection
# ---------------------------
def rebuild_timestamps(df):
    import pandas as _pd
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df['Date'] = df['Date'].astype(int)

    def parse_time_to_minutes(tstr):
        tstr = str(tstr).strip()
        for tf in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
            try:
                tt = datetime.strptime(tstr, tf).time()
                return tt.hour * 60 + tt.minute
            except Exception:
                continue
        raise ValueError(f"Unrecognized time format: '{tstr}'")

    minutes_of_day = df['Time'].apply(parse_time_to_minutes).astype(int).values

    # virtual day counter to detect month/rollover
    virtual_day = 0
    virtual_days = []
    prev_daynum = int(df.iloc[0]['Date'])
    virtual_days.append(virtual_day)
    for i in range(1, len(df)):
        curr_daynum = int(df.iloc[i]['Date'])
        if curr_daynum < prev_daynum:
            virtual_day += 1
        virtual_days.append(virtual_day)
        prev_daynum = curr_daynum
    virtual_days = np.array(virtual_days, dtype=int)

    abs_minutes_csv = virtual_days * 1440 + minutes_of_day
    diffs = np.diff(abs_minutes_csv, prepend=abs_minutes_csv[0])
    gap_threshold = 2 * FREQ_MINUTES
    gap_indices = np.where(diffs > gap_threshold)[0]

    # build sequential timestamps from first row
    first_time_str = str(df.iloc[0]['Time']).strip()
    parsed_time = None
    for tf in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
        try:
            parsed_time = datetime.strptime(first_time_str, tf).time()
            break
        except Exception:
            continue
    if parsed_time is None:
        raise ValueError(f"Could not parse start time: '{first_time_str}'")

    first_day = int(df.iloc[0]['Date'])
    try:
        start_ts = datetime(YEAR_PLACEHOLDER, 1, first_day, parsed_time.hour, parsed_time.minute, parsed_time.second)
    except Exception:
        start_ts = datetime(YEAR_PLACEHOLDER, 1, 1, parsed_time.hour, parsed_time.minute, parsed_time.second)
        print(f"[rebuild_timestamps] Warning: start day {first_day} invalid for month=1; using day=1 as start.")

    n = len(df)
    seq_timestamps = [start_ts + timedelta(minutes=FREQ_MINUTES * i) for i in range(n)]
    seq_ts_series = _pd.to_datetime(seq_timestamps)

    if len(gap_indices) > 0:
        print(f"[rebuild_timestamps] Detected {len(gap_indices)} gap(s) larger than {gap_threshold} minutes:")
        for gi in gap_indices:
            prev_idx = gi - 1 if gi - 1 >= 0 else 0
            gap_minutes = diffs[gi]
            csv_time_prev = f"{df.iloc[prev_idx]['Date']} {df.iloc[prev_idx]['Time']}"
            csv_time_curr = f"{df.iloc[gi]['Date']} {df.iloc[gi]['Time']}"
            print(f"  - gap at row {gi}: prev=({csv_time_prev}) -> curr=({csv_time_curr}), gap {gap_minutes} minutes")
            seq_ts_series.iloc[gi] = _pd.NaT
    else:
        print("[rebuild_timestamps] No large gaps detected.")

    df['timestamp'] = seq_ts_series.values
    df = df.sort_values('timestamp', na_position='last').reset_index(drop=True)
    n_nats = df['timestamp'].isna().sum()
    print(f"[rebuild_timestamps] Timestamps built. Range: {df['timestamp'].min()} -> {df['timestamp'].max()}. NaT count: {n_nats}")
    return df

# ---------------------------
# Feature engineering
# ---------------------------
def feature_engineering(df):
    df = df.copy()
    total_col_candidates = ['Total', 'TotalTraffic', 'Total Traffic', 'TotalTrafficCount']
    total_col = None
    for cand in total_col_candidates:
        if cand in df.columns:
            total_col = cand
            break
    if total_col is None:
        comps = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']
        if all(c in df.columns for c in comps):
            df['traffic_total'] = df['CarCount'] + df['BikeCount'] + df['BusCount'] + df['TruckCount']
            total_col = 'traffic_total'
        else:
            raise KeyError("Could not find Total traffic column and cannot reconstruct it. Columns: " + ", ".join(df.columns))

    if total_col != 'traffic_total':
        df = df.rename(columns={total_col: 'traffic_total'})

    # create time features; use ffill/bfill instead of deprecated fillna(method=...)
    df['hour'] = df['timestamp'].dt.hour.ffill().astype(int)
    df['minute'] = df['timestamp'].dt.minute.ffill().astype(int)
    df['dayofweek'] = df['timestamp'].dt.dayofweek.ffill().astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['month'] = df['timestamp'].dt.month.ffill().astype(int)
    df['day'] = df['timestamp'].dt.day.ffill().astype(int)

    # lag & rolling features
    df['traffic_lag_1'] = df['traffic_total'].shift(1)
    df['traffic_lag_4'] = df['traffic_total'].shift(SAMPLE_PER_HOUR)
    df['traffic_lag_24h'] = df['traffic_total'].shift(SAMPLE_PER_HOUR * 24)
    df['rolling_mean_1h'] = df['traffic_total'].rolling(window=SAMPLE_PER_HOUR, min_periods=1).mean().shift(1)
    df['rolling_mean_6h'] = df['traffic_total'].rolling(window=SAMPLE_PER_HOUR * 6, min_periods=1).mean().shift(1)
    df['rolling_std_6h'] = df['traffic_total'].rolling(window=SAMPLE_PER_HOUR * 6, min_periods=1).std().fillna(0).shift(1)

    # robust filling
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    return df

# ---------------------------
# Create future target
# ---------------------------
def create_future_target(df, steps_ahead=LOOKAHEAD_STEPS):
    df = df.copy()
    df['future_traffic'] = df['traffic_total'].shift(-steps_ahead)
    print("Future target created with steps_ahead=", steps_ahead)
    return df

# ---------------------------
# Prepare modeling data
# ---------------------------
def prepare_model_data(df):
    df = df.copy()
    drop_cols = ['Time', 'Date', 'Day of the week', 'Traffic Situation', 'month_inferred']
    for c in drop_cols:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    feature_cols = [
        'traffic_total', 'hour', 'minute', 'dayofweek', 'is_weekend', 'month',
        'traffic_lag_1', 'traffic_lag_4', 'traffic_lag_24h',
        'rolling_mean_1h', 'rolling_mean_6h', 'rolling_std_6h'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_model = df.dropna(subset=['future_traffic']).copy()
    X = df_model[feature_cols].astype(float)
    y = df_model['future_traffic'].astype(float)
    print("Prepared model data: X shape", X.shape, "y shape", y.shape)
    return X, y, feature_cols, df_model

# ---------------------------
# Chronological split
# ---------------------------
def chronological_train_test_split(X, y, test_ratio=TEST_RATIO):
    n = len(X)
    split_idx = int((1 - test_ratio) * n)
    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()
    print(f"Train/test split: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows")
    return X_train, X_test, y_train, y_test

# ---------------------------
# Train
# ---------------------------
def train_xgboost(X_train, y_train, X_val=None, y_val=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "verbosity": 1
    }
    num_round = 200
    evallist = [(dtrain, 'train')]
    if X_val_scaled is not None:
        deval = xgb.DMatrix(X_val_scaled, label=y_val)
        evallist.append((deval, 'eval'))
    model = xgb.train(params, dtrain, num_boost_round=num_round, evals=evallist, verbose_eval=25)
    return model, scaler

# ---------------------------
# Evaluate
# ---------------------------
def evaluate_model(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    dtest = xgb.DMatrix(X_test_scaled)
    preds = model.predict(dtest)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("Evaluation on test set:")
    print(f"  MSE: {mse:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  R2 : {r2:.3f}")
    return preds

# ---------------------------
# Map numeric prediction to label
# ---------------------------
def label_from_prediction(train_targets, pred_value):
    q1 = np.percentile(train_targets, 33)
    q2 = np.percentile(train_targets, 66)
    if pred_value <= q1:
        return "low"
    elif pred_value <= q2:
        return "normal"
    else:
        return "heavy"

# ---------------------------
# Main
# ---------------------------
def main():
    df = load_and_merge([CSV1, CSV2])
    df_ts = rebuild_timestamps(df)
    df_feat = feature_engineering(df_ts)
    df_target = create_future_target(df_feat, steps_ahead=LOOKAHEAD_STEPS)
    X, y, feature_cols, df_model = prepare_model_data(df_target)
    X_train, X_test, y_train, y_test = chronological_train_test_split(X, y, TEST_RATIO)

    val_split = int(0.95 * len(X_train))
    X_train_sub = X_train.iloc[:val_split]
    y_train_sub = y_train.iloc[:val_split]
    X_val = X_train.iloc[val_split:]
    y_val = y_train.iloc[val_split:]

    print("Training XGBoost...")
    model, scaler = train_xgboost(X_train_sub, y_train_sub, X_val, y_val)

    preds = evaluate_model(model, scaler, X_test, y_test)

    # Save
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols, "train_targets": y_train.values}, MODEL_FILE)
    print("Saved model and preprocessing to", MODEL_FILE)

    # Plot quick prediction vs actual (first N test rows)
    try:
        nplot = min(200, len(y_test))
        plt.figure(figsize=(12,4))
        plt.plot(range(nplot), y_test.values[:nplot], label='actual')
        plt.plot(range(nplot), preds[:nplot], label='predicted')
        plt.legend()
        plt.title("Prediction vs Actual (first {} test rows)".format(nplot))
        plt.xlabel("sample idx")
        plt.ylabel("traffic_total (predicted 48h ahead)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting failed:", e)

    # FEATURE IMPORTANCE (map f0->feature name)
    try:
        fmap = model.get_score(importance_type='gain')  # keys like 'f0','f1',...
        items = []
        for k,v in fmap.items():
            if k.startswith('f'):
                idx = int(k[1:])
                name = feature_cols[idx] if idx < len(feature_cols) else k
            else:
                name = k
            items.append((name, v))
        items = sorted(items, key=lambda x: x[1], reverse=True)
        if items:
            names, gains = zip(*items)
            plt.figure(figsize=(6, max(3, len(gains)*0.35)))
            plt.barh(range(len(gains)), gains[::-1])
            plt.yticks(range(len(gains)), names[::-1])
            plt.title("Feature importance (gain)")
            plt.tight_layout()
            plt.show()
        else:
            print("No feature importance available.")
    except Exception as e:
        print("Could not compute feature importance:", e)

    # ERROR BY HOUR
    try:
        residuals = y_test.values - preds
        df_eval = X_test.copy()
        df_eval['abs_err'] = np.abs(residuals)
        err_by_hour = df_eval.groupby('hour')['abs_err'].mean().sort_values(ascending=False)
        print("\nTop hours with largest avg absolute error:")
        print(err_by_hour.head(10))
    except Exception as e:
        print("Could not compute error-by-hour:", e)

    # EXAMPLE prediction using last row
    saved = joblib.load(MODEL_FILE)
    saved_model = saved['model']
    saved_scaler = saved['scaler']
    saved_feature_cols = saved['feature_cols']
    saved_train_targets = saved['train_targets']

    sample_row = X.iloc[-1:]
    scaled = saved_scaler.transform(sample_row)
    dmat = xgb.DMatrix(scaled)
    pred_val = saved_model.predict(dmat)[0]
    label = label_from_prediction(saved_train_targets, pred_val)
    print("\nExample prediction for last sample (48h ahead):", round(pred_val,1), "->", label)

    print("\nDone. Model saved to", MODEL_FILE)

if __name__ == "__main__":
    main()
