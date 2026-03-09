import sys, os, csv, json, torch, random, requests, threading, time, gc
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Konfiguracja - cofamy się o jeden poziom (z 'backend' do głównego folderu)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Wczytanie pliku config (zwróć uwagę na wcięcie w drugiej linijce)
with open(os.path.join(base_dir, 'config.json'), 'r') as f:
    CONFIG = json.load(f)

DEVICE = torch.device("cpu")
csv_file_lock = threading.Lock()
data_lock = threading.Lock()

# Ścieżka do zapisanych modeli
MODELS_DIR = os.path.join(base_dir, "saved_models")
os.makedirs(MODELS_DIR, exist_ok=True)



# === FUNKCJE POMOCNICZE (WYMAGANE PRZEZ INNE MODUŁY) ===

def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg) # Gunicorn to przechwyci i zapisze do app.log
    sys.stdout.flush()




def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def calculate_rsi(prices, period=14):
    series = pd.Series(prices)
    delta = series.diff()
    gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return (100 - (100 / (1 + rs))).fillna(50).values 

def calculate_volatility(prices, window=10):
    return pd.Series(prices).diff().rolling(window=window).std().fillna(0).values

def calculate_roc(prices, period=5):
    return (pd.Series(prices).pct_change(periods=period).fillna(0).values) * 100 

# === STATE & CACHE ===
class ServerState:
    STATUS = "STARTUP"
    LAST_PRICE = None
    MODEL_WEIGHTS = {"mc": 1.0, "rf": 1.0, "kan": 1.0, "net": 1.0}
    LAST_PREDICTIONS = {"mc": None, "rf": None, "kan": None, "net": None, "simdiff": None, "hakan": None}
    STRATEGY_MULT = 1.0 
    SIMDIFF_MULT = -1.0 
    HAKAN_MULT =  1.0 




# Rozszerzony bufor z funkcją bezpiecznego czyszczenia (Standard 2026)
CACHE = {
    "dates": [], 
    "history": [], 
    "forecast_dates": [], 
    "last_candle_ts": 0,
    "stoch": [], "trend": [], "res": [], "sup": [],
    "simdiff_curve": [], "hakan_curve": [],
    "prob_val": 50.0, "rf_prob_up": 50.0, "kan_val": 50.0, "neural_val": 50.0,
    "consensus_val": 50.0, "consensus_signal": "NEUTRAL",
    "simdiff_val": 50.0, "simdiff_signal": "NEUTRAL",
    "hakan_val": 50.0, "hakan_signal": "NEUTRAL",
    "pnl": {"times": [], "balance": []},
    "simdiff_pnl": {"times": [], "balance": []},
    "hakan_pnl": {"times": [], "balance": []},
    # Bufor RAM dla cen rzeczywistych (synchronizacja między kartami przeglądarki)
    "real_in_forecast_dates": [],
    "real_in_forecast_prices": [],
    "_real_last_ts": 0
}

def clear_training_buffers():
    """Czyści bufory przed nowym treningiem, zapobiegając wyciekom RAM."""
    with data_lock:
        CACHE["real_in_forecast_dates"] = []
        CACHE["real_in_forecast_prices"] = []
        CACHE["_real_last_ts"] = 0
        gc.collect() # Wymuszenie zwolnienia pamięci systemowej



# === PAPER TRADER (Z PEŁNĄ DATĄ, LOGAMI I OBLICZANIEM WSKAŹNIKÓW) ===
class PaperTrader:
    def __init__(self, filepath, initial_balance):
        self.filepath = filepath
        self.balance = initial_balance
        self.peak_balance = initial_balance  # Do obliczania Max Drawdown
        self.pnl_history = []                # Do obliczania Sharpe Ratio
        self.current_position = None
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
        with csv_file_lock:
            if not os.path.exists(self.filepath):
                with open(self.filepath, 'w', newline='') as f:
                    csv.writer(f).writerow(["Open_Time","Close_Time","Type","Size_BTC","Entry_Price","Exit_Price","PnL_USDT","Total_Balance","","Sharpe_Ratio","Max_Drawdown_%"])
            else:
                try:
                    df = pd.read_csv(self.filepath)
                    if not df.empty: 
                        self.balance = float(df.iloc[-1]["Total_Balance"])
                        self.peak_balance = float(df["Total_Balance"].max()) # Odtwarzamy historyczny szczyt konta
                        
                        # Odtwarzamy historię PnL, pomijając ewentualne błędy w danych
                        if "PnL_USDT" in df.columns:
                            self.pnl_history = pd.to_numeric(df["PnL_USDT"], errors='coerce').dropna().tolist()
                except Exception as e: 
                    log_msg(f"PaperTrader Init Err: {e}")

    def open_position(self, direction, entry_price, entry_time_str):
        self.current_position = {'type': direction, 'entry': entry_price, 'time': entry_time_str}
        log_msg(f"+++ [{os.path.basename(self.filepath)}] OPEN {direction} at {entry_price:.2f}")

    def close_position(self, exit_price, exit_time_str):
        if not self.current_position: return
        p = self.current_position
        
        # Obliczanie PnL
        if p['type'] == 'LONG':
            pnl = (exit_price - p['entry']) * CONFIG["Position_Size_Btc"] 
        else:
            pnl = (p['entry'] - exit_price) * CONFIG["Position_Size_Btc"]
            
        self.balance += pnl
        self.pnl_history.append(pnl)

        # 1. OBLICZANIE MAX DRAWDOWN (%)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            
        drawdown_pct = 0.0
        if self.peak_balance > 0:
            drawdown_pct = ((self.peak_balance - self.balance) / self.peak_balance) * 100.0

        # 2. OBLICZANIE WSKAŹNIKA SHARPE'A (Uproszczony, na podstawie transakcji)
        sharpe_ratio = 0.0
        if len(self.pnl_history) > 1:
            mean_pnl = np.mean(self.pnl_history)
            std_pnl = np.std(self.pnl_history)
            if std_pnl > 0:
                sharpe_ratio = mean_pnl / std_pnl # Pomijamy stopę wolną od ryzyka dla prostoty w krypto

        log_msg(f"--- [{os.path.basename(self.filepath)}] CLOSE {p['type']} at {exit_price:.2f} | PnL: {pnl:.2f} | Bal: {self.balance:.2f} | DD: {drawdown_pct:.2f}% | Sharpe: {sharpe_ratio:.2f}")
        
        # Zapis do CSV (z faktycznymi obliczeniami zamiast zer)
        with csv_file_lock:
            with open(self.filepath, 'a', newline='') as f:
                csv.writer(f).writerow([
                    p['time'], 
                    exit_time_str, 
                    p['type'], 
                    CONFIG["Position_Size_Btc"], 
                    f"{p['entry']:.2f}", 
                    f"{exit_price:.2f}", 
                    f"{pnl:.2f}", 
                    f"{self.balance:.2f}", 
                    "", 
                    f"{sharpe_ratio:.3f}",   # Wpisuje obliczony Sharpe Ratio (3 miejsca po przecinku)
                    f"{drawdown_pct:.2f}"    # Wpisuje obliczony Drawdown % (2 miejsca po przecinku)
                ])
                
        self.current_position = None




# === DATA FETCHING ===

def get_data_server():
    limit = CONFIG["Total_Data_Points"] + 100 
    required_min = CONFIG["Lookback_Window"] + CONFIG["Future_Prediction_Steps"] + 50
    
    for attempt in range(60): # 60 prób
        try:
            r = requests.get("https://api.binance.com/api/v3/klines", 
                             params={"symbol":"BTCUSDT", "interval":"1m", "limit": limit}, 
                             timeout=5).json()
            
            if not isinstance(r, list) or len(r) < required_min:
                log_msg(f"Ostrzeżenie: Binance zwrócił za mało danych ({len(r)}/{required_min}). Próba {attempt+1}...")
                time.sleep(2)
                continue
                
            df = pd.DataFrame(r).iloc[:-1] # Usuwamy ostatnią niedomkniętą świecę
            
            # Sprawdzenie ciągłości danych (czy nie ma luk czasowych powyżej 1 minuty)
            ts = df[0].astype(int).values
            diffs = np.diff(ts)
            if np.any(diffs > 60000): # 60000 ms = 1 min
                log_msg("Krytyczna luka w danych Binance! Ponawiam zapytanie...")
                time.sleep(2)
                continue

            return [datetime.fromtimestamp(t/1000) for t in ts], df[4].astype(float).values, ts.tolist()
            
        except Exception as e:
            log_msg(f"Błąd pobierania danych (Próba {attempt+1}): {e}")
            time.sleep(2)
            
    log_msg("BŁĄD KRYTYCZNY: Nie udało się pobrać stabilnych danych po 60 próbach.")
    return [], [], []


def prepare_data(prices):
    rsi, vol, roc = calculate_rsi(prices), calculate_volatility(prices), calculate_roc(prices)
    diffs = np.zeros_like(prices); diffs[1:] = np.diff(prices)
    sd, sr, sv, src = StandardScaler(), MinMaxScaler((-1,1)), MinMaxScaler((0,1)), StandardScaler()
    ds = np.hstack((sd.fit_transform(diffs.reshape(-1,1)), sr.fit_transform(rsi.reshape(-1,1)), sv.fit_transform(vol.reshape(-1,1)), src.fit_transform(roc.reshape(-1,1))))
    w, h = CONFIG["Lookback_Window"], CONFIG["Future_Prediction_Steps"]
    X, y, yh = [], [], []
    for i in range(len(ds)-w-h):
        X.append(ds[i:i+w]); y.append(ds[i+w, 0]); yh.append(ds[i+w:i+w+h, 0])
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32), torch.tensor(np.array(yh), dtype=torch.float32), sd, sr, sv, src, prices


TRADER = PaperTrader(os.path.join(base_dir, CONFIG["Pnl_File_Path"]), CONFIG["Initial_Balance"])
SIMDIFF_TRADER = PaperTrader(os.path.join(base_dir, CONFIG["SimDiff_Pnl_File"]), CONFIG["Initial_Balance"])
HAKAN_TRADER = PaperTrader(os.path.join(base_dir, CONFIG.get("HaKAN_Pnl_File", "static/hakan_pnl.csv")), CONFIG["Initial_Balance"])