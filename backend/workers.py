import numpy as np
import torch

from backend.utils import CONFIG, DEVICE, log_msg

# Importy modeli (przeniesione z app.py)
from backend.models.neural_trend_model import DualModeNetwork, train_dual_mode_model
from backend.models.monte_carlo_model import calculate_monte_carlo_probability, forecast
from backend.models.random_forest_model import run_random_forest
from backend.models.kan_model import train_and_predict_kan

from backend.models.simdiffdit_model import SimDiffHorizonModel, train_simdiff, run_simdiff_inference
from backend.models.hakan_model import HaKANModel, train_hakan, run_hakan_inference

''' 
worker_ensemble - to jest strategia "Main Consensus". W tym jednym procesie po kolei uruchamiają się aż 4 modele: Neural Trend, Monte Carlo, Random Forest i KAN. Ich wyniki są na bieżąco łączone w jedną wspólną decyzję. 
'''
def worker_ensemble(q, raw, X, y, win, sd, sr, sv, src):
    try:
        net = DualModeNetwork(4, CONFIG["Hidden_Dim"], CONFIG["Num_Layers"]).to(DEVICE)
        train_dual_mode_model(net, X, y)
        
        f = forecast(net, win, CONFIG["Future_Prediction_Steps"], sd, sr, sv, src, raw[-1], raw)
        
        if isinstance(f, (list, tuple)) and len(f) >= 2:
            ts = (1/(1+np.exp(-((f[1][-1]-raw[-1])/raw[-1]*100)*50)))*100
        else:
            ts = 50.0 
            log_msg("Warning: Forecast zwrócił nieprawidłowy format. Ustawiam neutralny trend.")
            
        log_msg(f">>> [1/4] Neural Trend: {ts:.2f}%")
        
        mc = calculate_monte_carlo_probability(net, win, sd, sr, sv, src, raw[-1], raw)
        log_msg(f">>> [2/4] Monte Carlo: {mc:.2f}%")
        
        rf = run_random_forest(raw)
        kan = train_and_predict_kan(X, y, win)
        
        q.put({'mc_prob': mc, 'rf': rf, 'kan_prob': kan, 'trend_score': ts, 'forecast':[list(x) for x in f]})
    except Exception as e: 
        log_msg(f"CRITICAL Ens Err: {e}")
        q.put(None) 


def worker_simdiff(q, raw, X, yh, win, sd, sr, sv, src):
    try:
        m = SimDiffHorizonModel(4, 128, 4, 4, CONFIG["Lookback_Window"], CONFIG["Future_Prediction_Steps"]).to(DEVICE)
        train_simdiff(m, X, yh)
        p = run_simdiff_inference(m, win, CONFIG["Future_Prediction_Steps"], sd, sr, sv, src, raw[-1])
        sc = (1/(1+np.exp(-((p[-1]-raw[-1])/raw[-1]*100)*2.5)))*100
        
        log_msg(f">>> [SimDiff] Score: {sc:.2f}%")
        q.put({'score': sc, 'preds': list(p)})
    except Exception as e: 
        log_msg(f"SimDiff Worker Err: {e}") 
        q.put(None)


def worker_hakan(q, raw, X, yh, win, sd):
    try:
        m = HaKANModel(
            input_dim=CONFIG["Lookback_Window"] * 4, 
            hidden_dim=CONFIG.get("HaKAN_Hidden", 128), 
            pred_len=CONFIG["Future_Prediction_Steps"]
        ).to(DEVICE)
        
        # Cała brudna robota jest już ukryta wewnątrz train_hakan()
        train_hakan(m, X, yh)
        
        preds = run_hakan_inference(m, win, CONFIG["Future_Prediction_Steps"], sd, raw[-1])
        
        hakan_sens = CONFIG.get("HaKAN_Sensitivity", 1.5)
        score = (1 / (1 + np.exp(-((preds[-1] - raw[-1]) / raw[-1] * 100) * hakan_sens))) * 100

        log_msg(f">>>[HaKAN] Score: {score:.2f}%")
        
        q.put({'score': score, 'preds': preds.tolist()})
    except Exception as e:
        log_msg(f"HaKAN Worker Err: {e}")
        q.put(None)