import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from backend.utils import CONFIG, calculate_rsi, calculate_roc, calculate_volatility, log_msg, MODELS_DIR

def prepare_rf_data(prices):
    df = pd.DataFrame({"close": prices})
    df['rsi'] = calculate_rsi(prices, CONFIG["Rf_Rsi_Period"])
    df['roc'] = calculate_roc(prices, CONFIG["Rf_Roc_Period"])
    df['vol'] = calculate_volatility(prices, CONFIG["Rf_Volatility_Window"])
    
    # Bollinger Bands
    df['sma_bb'] = df['close'].rolling(window=CONFIG["Rf_Bb_Window"]).mean()
    std_dev = df['close'].rolling(window=CONFIG["Rf_Bb_Window"]).std()
    df['bb_position'] = (df['close'] - df['sma_bb']) / (CONFIG["Rf_Bb_Std"] * std_dev + 1e-9)
    df['dist_sma'] = (df['close'] - df['sma_bb']) / (df['sma_bb'] + 1e-9)
    
    # Naprawa dla nowszych wersji Pandas (zastępuje fillna(method='bfill'))
    df = df.replace([np.inf, -np.inf], np.nan).bfill().ffill()
    
    df['future_close'] = df['close'].shift(-CONFIG["Rf_Lookahead"])
    df['target'] = (df['future_close'] > df['close']).astype(int)
    
    features = ['rsi', 'roc', 'vol', 'dist_sma', 'bb_position']
    train_df = df.dropna(subset=['target'])
    
    if len(train_df) < 20:
        return None, None, None

    X = train_df[features].values
    y = train_df['target'].values
    last_row = df[features].iloc[-1].values.reshape(1, -1)
    
    return X, y, last_row

def run_random_forest(raw_all_prices):
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    rf_X, rf_y, rf_last_row = prepare_rf_data(raw_all_prices)
    
    if rf_X is None:
        return 50.0, 50.0, 0.5, 0.0, 0.0

    split = int(len(rf_X) * 0.8)
    X_train, X_test, y_train, y_test = rf_X[:split], rf_X[split:], rf_y[:split], rf_y[split:]
    
    # Zwiększamy nieco głębokość drzew dla lepszej czułości
    rf_model = RandomForestClassifier(n_estimators=CONFIG["Rf_Estimators"], max_depth=7, random_state=42)

    try:
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, y_pred)
    except:
        rf_acc = 0.5

    rf_model.fit(rf_X, rf_y)
    probs = rf_model.predict_proba(rf_last_row)[0]
    rf_raw_prob_float = probs[1] * 100.0 
    
    try: joblib.dump(rf_model, model_path)
    except: pass
    
    # --- LOGIKA SWING (AGRESYWNA) ---
    raw_diff = rf_raw_prob_float - 50.0
    
    # Zamiast blokować model, gdy Acc < 0.5, dajemy mu stałą wagę.
    # Dzięki temu model zawsze "macha" wynikiem, nawet gdy słabo zgaduje.
    weight = 0.8 + (max(0, rf_acc - 0.5) * 0.4)
    
    rf_final_val = 50.0 + (raw_diff * weight)
    rf_final_val = max(1.0, min(99.0, rf_final_val))
    
    log_msg(f">>> [3/4] Random Forest: {rf_final_val:.2f}% (Raw: {rf_raw_prob_float:.1f}%, Acc: {rf_acc:.3f})")
    
    # ZWRACANE WARTOŚCI (Dla Twojego opisu na stronie):
    # 1. rf_final_val (Wartość po strzałce ➜)
    # 2. rf_raw_prob_float (Wartość Raw:)
    # 3. rf_acc (Wartość Acc:)
    # 4. raw_diff (Różnica, np. -20.8)
    # 5. weight (Współczynnik Trust)
    return round(rf_final_val, 2), round(rf_raw_prob_float, 2), round(rf_acc, 3), round(raw_diff, 2), round(weight, 2)
