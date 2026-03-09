import sys, os, time, copy, io, threading, multiprocessing, requests, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ZABEZPIECZENIE: Zapewnienie, że modele AI używają bezpiecznej metody 'spawn'
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from flask import Flask, jsonify, render_template, send_file, request
from flask_cors import CORS
from datetime import datetime, timedelta

# Importy z utils
from backend.utils import CONFIG, DEVICE, ServerState, CACHE, TRADER, SIMDIFF_TRADER, HAKAN_TRADER
from backend.utils import log_msg, set_seed, cleanup_memory, MODELS_DIR, get_data_server, prepare_data, data_lock, csv_file_lock, calculate_rsi, calculate_volatility, calculate_roc
from backend.utils import clear_training_buffers

# IMPORTUJEMY TYLKO WORKERY (żadnego mieszania w wagach i modelach w app.py!)
from backend.workers import worker_ensemble, worker_simdiff, worker_hakan

app = Flask(__name__, template_folder='frontend/templates', static_folder='frontend/static')
CORS(app)




# ==========================================================
# LOGOWANIE AKTYWNOŚCI UŻYTKOWNIKÓW WWW DO OSOBNEGO PLIKU
# ==========================================================
@app.before_request
def log_visitor_activity():
    # 1. Pobieranie prawdziwego IP zza Nginx/Cloudflare
    ip = request.headers.get('X-Forwarded-For', request.remote_addr).split(',')[0].strip()
    path = request.path
    
    # 2. Nie logujemy zapytań systemowych bota (/api/) ani plików stylów (/static/)
    if not path.startswith('/api/') and not path.startswith('/static/'):
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_agent = request.headers.get('User-Agent', '')
        
        # 3. Zapisujemy ruch tylko jeśli to człowiek, a nie nasz "samowyzwalacz"
        if "python-requests" not in user_agent:
            log_line = f"[{now_str}] Wejście z IP: {ip} | URL: {path} | Klient: {user_agent}\n"
            
            try:
                with open("visitors_activity.log", "a", encoding="utf-8") as f:
                    f.write(log_line)
            except:
                pass





def adjust_weights(current_price):
    if ServerState.LAST_PRICE is None or ServerState.LAST_PREDICTIONS["mc"] is None: return
    actual_up = current_price > ServerState.LAST_PRICE
    for name, prev in ServerState.LAST_PREDICTIONS.items():
        if name in['simdiff', 'hakan'] or prev is None: continue
        is_correct = (prev > 50.0) == actual_up
        ServerState.MODEL_WEIGHTS[name] = max(0.2, min(2.0, ServerState.MODEL_WEIGHTS[name] + (0.05 if is_correct else -0.05)))





def run_ai_training_sequence():
    try:
        start_time_perf = time.time()
        ServerState.STATUS = "TRAINING"

        clear_training_buffers()

        times, prices, ts = get_data_server()
        if not times: 
            ServerState.STATUS = "READY"
            return

        current_minute_ts = int(time.time() // 60) * 60
        if ts[-1] / 1000 >= current_minute_ts:
            times = times[:-1]
            prices = prices[:-1]
            ts = ts[:-1]

        try: 
            ex_p_close = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=3).json()['price'])
        except: 
            ex_p_close = prices[-1]
        
        ex_t_close = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for t_obj in[TRADER, SIMDIFF_TRADER, HAKAN_TRADER]:
            if t_obj.current_position: 
                t_obj.close_position(ex_p_close, ex_t_close)
                
        adjust_weights(ex_p_close)

        X, y, yh, sd, sr, sv, src, raw = prepare_data(prices)
        
        def mk_win():
            d = sd.transform(np.diff(raw).reshape(-1,1)[-CONFIG["Lookback_Window"]:])
            r = sr.transform(calculate_rsi(raw)[-CONFIG["Lookback_Window"]:].reshape(-1,1))
            v = sv.transform(calculate_volatility(raw)[-CONFIG["Lookback_Window"]:].reshape(-1,1))
            rc = src.transform(calculate_roc(raw)[-CONFIG["Lookback_Window"]:].reshape(-1,1))
            return np.hstack((d, r, v, rc))
        win = mk_win()

        q1, q2, q3 = multiprocessing.Queue(), multiprocessing.Queue(), multiprocessing.Queue()
        p1 = multiprocessing.Process(target=worker_ensemble, args=(q1, raw, X, y, win, sd, sr, sv, src))
        p2 = multiprocessing.Process(target=worker_simdiff, args=(q2, raw, X, yh, win, sd, sr, sv, src))
        p3 = multiprocessing.Process(target=worker_hakan, args=(q3, raw, X, yh, win, sd))
        
        p1.start(); p2.start(); p3.start()
        
        def safe_get(q, process):
            while process.is_alive():
                try: return q.get(timeout=1)
                except: pass
            try: return q.get(timeout=1) 
            except: return None

        r1 = safe_get(q1, p1)
        r2 = safe_get(q2, p2)
        r3 = safe_get(q3, p3)
        
        p1.join(); p2.join(); p3.join()

        if r1 is None or r2 is None or r3 is None:
            log_msg("CRITICAL: Jeden z modeli uległ awarii. Przerywam.")
            ServerState.STATUS = "ERROR"
            return

        duration = time.time() - start_time_perf
        log_msg(f">>> [FINISH] Trening wszystkich modeli zakończony w {duration:.2f}s.")

        try: 
            ex_p_open = float(requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=3).json()['price'])
        except: 
            ex_p_open = raw[-1]
        
        ex_t_open = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with data_lock:
            m1_val = ServerState.STRATEGY_MULT if CONFIG["Consensus_Mode"]==2 else ServerState.STRATEGY_MULT
            m2_val = ServerState.SIMDIFF_MULT if CONFIG["SimDiff_Mode"]==2 else ServerState.SIMDIFF_MULT
            m3_val = ServerState.HAKAN_MULT if CONFIG.get("HaKAN_Mode", 2)==2 else ServerState.HAKAN_MULT

            if isinstance(r1['rf'], (list, tuple)):
                rf_v, rf_raw, rf_acc, rf_diff, rf_t = r1['rf']
            else:
                rf_v, rf_raw, rf_acc, rf_diff, rf_t = r1['rf'], 50.0, 0.5, 0.0, 0.0

            CACHE["prob_val"], CACHE["kan_val"], CACHE["neural_val"] = r1['mc_prob'], r1['kan_prob'], r1['trend_score']
            CACHE["rf_prob_up"], CACHE["rf_raw_prob"], CACHE["rf_acc_view"] = rf_v, rf_raw, rf_acc
            CACHE["rf_calc_steps"] = {"diff": rf_diff, "trust": rf_t}
            
            w = ServerState.MODEL_WEIGHTS
            avg = (r1['mc_prob']*w['mc'] + rf_v*w['rf'] + r1['kan_prob']*w['kan'] + r1['trend_score']*w['net']) / sum(w.values())
            
            CACHE["consensus_val"] = round(avg, 2)
            CACHE["consensus_signal"] = "BUY" if avg > 50.0 else ("SELL" if avg < 50.0 else "NEUTRAL")
            CACHE["strategy_mult_cache"] = m1_val

            CACHE["simdiff_val"], CACHE["simdiff_curve"] = round(r2['score'],1), [raw[-1]] + r2['preds']
            CACHE["simdiff_signal"] = "BUY" if r2['score'] > 50.0 else ("SELL" if r2['score'] < 50.0 else "NEUTRAL")
            CACHE["simdiff_mult_cache"] = m2_val

            CACHE["hakan_val"], CACHE["hakan_curve"] = round(r3['score'],1),[raw[-1]] + r3['preds']
            CACHE["hakan_signal"] = "BUY" if r3['score'] > 50.0 else ("SELL" if r3['score'] < 50.0 else "NEUTRAL")
            CACHE["hakan_mult_cache"] = m3_val

            CACHE["history"] = raw[-CONFIG["History_Show"]:].tolist()
            CACHE["dates"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times[-CONFIG["History_Show"]:]]
            CACHE["forecast_dates"] = [(times[-1] + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(CONFIG["Future_Prediction_Steps"]+1)]
            CACHE["last_candle_ts"] = ts[-1]
            
            f = r1['forecast']
            v_now = np.std(raw[-15:])
            CACHE["stoch"], CACHE["trend"] = [raw[-1]]+f[0], [raw[-1]]+f[1]
            CACHE["res"], CACHE["sup"] =[raw[-1]+v_now*2]+f[2], [raw[-1]-v_now*2]+f[3]

            dir_cons = "NEUTRAL" if CACHE["consensus_signal"] == "NEUTRAL" else ("LONG" if (avg - 50.0) * m1_val > 0 else "SHORT")
            dir_sim  = "NEUTRAL" if CACHE["simdiff_signal"] == "NEUTRAL" else ("LONG" if (r2['score'] - 50.0) * m2_val > 0 else "SHORT")
            dir_hak  = "NEUTRAL" if CACHE["hakan_signal"] == "NEUTRAL" else ("LONG" if (r3['score'] - 50.0) * m3_val > 0 else "SHORT")

            is_hedge_enabled = CONFIG.get("Anti_Herd_Enable", True)

            if is_hedge_enabled and (dir_cons == dir_sim == dir_hak) and (dir_cons != "NEUTRAL"):
                confidences = {
                    "Main": abs(avg - 50.0),
                    "SimDiff": abs(r2['score'] - 50.0),
                    "HaKAN": abs(r3['score'] - 50.0)
                }
                weakest_model = min(confidences, key=confidences.get)
                flipped_dir = "SHORT" if dir_cons == "LONG" else "LONG"
                
                log_msg(f"*** ANTI-HERD: Stado gra {dir_cons}. Odwracam pozycję '{weakest_model}' na {flipped_dir} jako HEDGE.")

                if weakest_model == "Main":
                    dir_cons = flipped_dir
                    CACHE["consensus_signal"] = "SELL (HEDGE)" if "BUY" in CACHE["consensus_signal"] else "BUY (HEDGE)"
                elif weakest_model == "SimDiff":
                    dir_sim = flipped_dir
                    CACHE["simdiff_signal"] = "SELL (HEDGE)" if "BUY" in CACHE["simdiff_signal"] else "BUY (HEDGE)"
                elif weakest_model == "HaKAN":
                    dir_hak = flipped_dir
                    CACHE["hakan_signal"] = "SELL (HEDGE)" if "BUY" in CACHE["hakan_signal"] else "BUY (HEDGE)"

            if dir_cons != "NEUTRAL": TRADER.open_position(dir_cons, ex_p_open, ex_t_open)
            if dir_sim != "NEUTRAL": SIMDIFF_TRADER.open_position(dir_sim, ex_p_open, ex_t_open)
            if dir_hak != "NEUTRAL": HAKAN_TRADER.open_position(dir_hak, ex_p_open, ex_t_open)

            ServerState.LAST_PRICE = ex_p_open
            ServerState.LAST_PREDICTIONS = {
                "mc": r1['mc_prob'], "rf": rf_v, "kan": r1['kan_prob'], 
                "net": r1['trend_score'], "simdiff": r2['score'], "hakan": r3['score']
            }
            
            if CONFIG["Consensus_Mode"]==2: ServerState.STRATEGY_MULT *= -1
            if CONFIG["SimDiff_Mode"]==2: ServerState.SIMDIFF_MULT *= -1
            if CONFIG.get("HaKAN_Mode", 2)==2: ServerState.HAKAN_MULT *= -1

        ServerState.STATUS = "READY"
    except Exception as e: 
        log_msg(f"Sequence Err: {e}")
        ServerState.STATUS = "ERROR"


@app.route('/api/init')
def api_init():
    try:
        last_training_ts = CACHE.get("last_candle_ts", 0)
        if last_training_ts > 0:
            r = requests.get("https://api.binance.com/api/v3/klines", params={"symbol": "BTCUSDT", "interval": "1m", "limit": 10}, timeout=2).json()
            max_allowed_ts = last_training_ts + (CONFIG["Future_Prediction_Steps"] * 60 * 1000)
            
            with data_lock:
                for k in r:
                    c_ts = int(k[0])
                    if last_training_ts < c_ts <= max_allowed_ts:
                        c_time = datetime.fromtimestamp(c_ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
                        
                        if c_time not in CACHE["real_in_forecast_dates"]:
                            CACHE["real_in_forecast_dates"].append(c_time)
                            CACHE["real_in_forecast_prices"].append(float(k[4]))
                            if c_ts > CACHE.get("_real_last_ts", 0):
                                CACHE["_real_last_ts"] = c_ts
                
                if CACHE["real_in_forecast_dates"]:
                    combined = list(zip(CACHE["real_in_forecast_dates"], CACHE["real_in_forecast_prices"]))
                    combined.sort(key=lambda x: x[0])
                    CACHE["real_in_forecast_dates"] = [x[0] for x in combined]
                    CACHE["real_in_forecast_prices"] = [x[1] for x in combined]
    except:
        pass

    with data_lock:
        merged_history = list(CACHE["history"]) + list(CACHE["real_in_forecast_prices"])
        merged_dates =[str(d) for d in CACHE["dates"]] + [str(d) for d in CACHE["real_in_forecast_dates"]]

        d = {
            "history": merged_history,
            "dates": merged_dates,
            "forecast_dates": CACHE["forecast_dates"],
            "stoch": CACHE["stoch"],
            "trend": CACHE["trend"],
            "res": CACHE["res"],
            "sup": CACHE["sup"],
            "simdiff_curve": CACHE["simdiff_curve"],
            "hakan_curve": CACHE["hakan_curve"],
            "consensus_signal": CACHE["consensus_signal"],
            "simdiff_signal": CACHE["simdiff_signal"],
            "hakan_signal": CACHE["hakan_signal"]
        }
        
        for k, obj in[("pnl", TRADER), ("simdiff_pnl", SIMDIFF_TRADER), ("hakan_pnl", HAKAN_TRADER)]:
            try:
                df = pd.read_csv(obj.filepath)
                d[k] = df[["Close_Time", "Total_Balance"]].values.tolist()
            except: d[k] =[]

        d['models'] = {
            'mc_prob': CACHE['prob_val'], 
            'rf_prob': CACHE['rf_prob_up'], 
            'rf_raw': CACHE.get('rf_raw_prob', '--'),
            'rf_acc': CACHE.get('rf_acc_view', '--'),
            'kan_prob': CACHE['kan_val'], 
            'neural_prob': CACHE['neural_val'],
            'simdiff_prob': CACHE['simdiff_val'], 
            'hakan_val': CACHE['hakan_val'], 
            'consensus_val': CACHE['consensus_val'], 
            'weights': ServerState.MODEL_WEIGHTS,
            'mult': CACHE.get('strategy_mult_cache', 1.0),
            'simdiff_mult': CACHE.get('simdiff_mult_cache', 1.0),
            'hakan_mult': CACHE.get('hakan_mult_cache', 1.0),
            'config': {
                "win": CONFIG["Lookback_Window"], 
                "ahead": CONFIG["Future_Prediction_Steps"],
                "iter": CONFIG.get("Mc_Iterations", 50),
                "t": CONFIG.get("Mc_Temp_Mult", 9.0),
                "s": CONFIG.get("Mc_Trend_Suppression", 0.3)
            }
        }

    return jsonify(d)


@app.route('/api/current_price')
def api_current():
    try:
        r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=2).json()
        p = float(r['price'])
        k = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1", timeout=2).json()[0]

        candle_ts = int(k[0])
        candle_time = datetime.fromtimestamp(candle_ts / 1000).strftime("%Y-%m-%d %H:%M:%S")
        candle_price = float(k[4])

        with data_lock:
            last_training_ts = CACHE.get("last_candle_ts", 0)
            max_allowed_ts = last_training_ts + (CONFIG["Future_Prediction_Steps"] * 60 * 1000)

            if last_training_ts > 0 and candle_ts > last_training_ts:
                if candle_ts <= max_allowed_ts:
                    if candle_ts == CACHE.get("_real_last_ts", 0):
                        if CACHE["real_in_forecast_prices"]:
                            CACHE["real_in_forecast_prices"][-1] = candle_price
                    elif candle_ts > CACHE.get("_real_last_ts", 0):
                        CACHE["real_in_forecast_dates"].append(candle_time)
                        CACHE["real_in_forecast_prices"].append(candle_price)
                        CACHE["_real_last_ts"] = candle_ts
                else:
                    if CACHE["real_in_forecast_prices"]:
                        CACHE["real_in_forecast_prices"][-1] = candle_price

        return jsonify({
            "price": p, "status": ServerState.STATUS, 
            "closed_candle": {
                "time": candle_time,
                "price": candle_price, "ts": candle_ts
            }
        })
    except: return jsonify({"status": "OFFLINE"})


@app.route('/download-csv')
def dl1(): return send_file(TRADER.filepath, as_attachment=True)
@app.route('/download-simdiff-csv')
def dl2(): return send_file(SIMDIFF_TRADER.filepath, as_attachment=True)
@app.route('/download-hakan-csv')
def dl3(): return send_file(HAKAN_TRADER.filepath, as_attachment=True)
@app.route('/')
def index(): return render_template('index.html')


# ==========================================================
# GŁÓWNA PĘTLA BOTA
# ==========================================================
def background_worker():
    time.sleep(1) 
    while True:
        cycle_start_time = time.time()
        run_ai_training_sequence()
        
        now = datetime.now()
        steps = CONFIG.get("Future_Prediction_Steps", 5)
        minutes_to_next = steps - (now.minute % steps)
        target_time = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_next, seconds=2)
        
        wait_seconds = (target_time - datetime.now()).total_seconds()
        
        if wait_seconds <= 0: wait_seconds += (steps * 60)
            
        train_duration = time.time() - cycle_start_time
        log_msg(f"Cykl trwał: {train_duration:.2f}s.")
        log_msg(f"Zegar: Następny trening o {target_time.strftime('%H:%M:%S')}. Śpię {wait_seconds:.0f}s.")
        time.sleep(wait_seconds)


# ==========================================================
# INICJALIZACJA - AUTOMATYCZNY START BEZ CZEKANIA NA UŻYTKOWNIKA
# ==========================================================
_bot_started_lock = threading.Lock()
_bot_started = False

# 1. Główny włącznik sprzęgnięty z pierwszym zapytaniem (najbezpieczniejsza metoda we Flasku)
@app.before_request
def start_bot_once():
    global _bot_started
    if not _bot_started:
        with _bot_started_lock:
            if not _bot_started:
                t = threading.Thread(target=background_worker, daemon=True)
                t.start()
                _bot_started = True
                log_msg(">>> Background Worker został uruchomiony automatycznie (Auto-Ping).")


# 2. Samowyzwalacz: Wątek, który po prostu odczekuje 3 sekundy i sam ładuje stronę
# Dzięki temu nie musisz ręcznie uruchamiać przeglądarki!
def auto_ping_server():
    time.sleep(3)
    try: 
        # Robi "puste" zapytanie żeby wybudzić aplikację po jej starcie
        requests.get("http://127.0.0.1:8052/api/current_price", timeout=5)
    except: 
        pass

# Uruchamia samowyzwalacz, ale chroni przed wybuchem wątków (Thread Bomb)
if multiprocessing.current_process().name == 'MainProcess':
    threading.Thread(target=auto_ping_server, daemon=True).start()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8052)