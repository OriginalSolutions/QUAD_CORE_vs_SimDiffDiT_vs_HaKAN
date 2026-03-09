import torch
import numpy as np
import random
from backend.utils import CONFIG, DEVICE, calculate_rsi, log_msg

def forecast(model, start_win, steps, s_diff, s_rsi, s_vol, s_roc, last_price_val, history_prices, temp_override=None, suppression=None):
    model.eval()
    curr_input = torch.tensor(start_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    current_temp = temp_override if temp_override is not None else CONFIG["Temperature"]
    is_monte_carlo = temp_override is not None and temp_override > 1.0
    trend_suppression = suppression if suppression is not None else 1.0
    preds_stoch, preds_trend, preds_res, preds_sup = [], [], [], []
    sim_history = [last_price_val] * (CONFIG["Rsi_Period"] + 20) 
    recent_volatility = np.std(history_prices[-15:]) if len(history_prices) > 15 else last_price_val * 0.005
    initial_offset_res = recent_volatility * 2.0
    initial_offset_sup = recent_volatility * 2.0
    start_jitter = 0
    if is_monte_carlo: start_jitter = np.random.normal(0, recent_volatility * 0.2)
    pos_trend = last_price_val + start_jitter
    vel_trend = 0.0; pos_stoch = last_price_val + start_jitter; vel_stoch = 0.0
    pos_res = last_price_val + initial_offset_res; vel_res = 0.0
    pos_sup = last_price_val - initial_offset_sup; vel_sup = 0.0
    INERTIA_TREND = 0.60; FORCE_TREND = 0.25; MAX_VELOCITY = recent_volatility * 0.9 
    INERTIA_STOCH = 0.50; FORCE_STOCH = 0.70
    TETHER_STRENGTH = 0.0015 * (abs(trend_suppression) if trend_suppression < 1.0 else 1.0)
    drift_res_factor = random.uniform(0.9, 1.1)
    drift_sup_factor = random.uniform(0.9, 1.1)
    with torch.no_grad():
        for i in range(steps):
            mu, sigma, quantiles = model(curr_input)
            raw_mu = mu.item(); raw_sigma = sigma.item()
            real_trend_delta = s_diff.inverse_transform([[raw_mu]])[0][0]
            real_sigma_delta = raw_sigma * s_diff.scale_[0] 
            if abs(trend_suppression) < 0.1: avg_drift = s_diff.mean_[0]; real_trend_delta -= avg_drift
            trend_component = real_trend_delta * trend_suppression
            noise_component = 0.0
            if is_monte_carlo: noise_component = real_sigma_delta * torch.randn(1).item() * current_temp
            final_delta_usd = trend_component + noise_component
            max_allowed_move = recent_volatility * 3.0
            delta_trend = np.clip(final_delta_usd, -max_allowed_move, max_allowed_move)
            target_trend = pos_trend + delta_trend
            raw_width_res = quantiles[0, 0].item() * s_diff.scale_[0] * 8.0
            raw_width_sup = quantiles[0, 1].item() * s_diff.scale_[0] * 8.0
            current_rsi_norm = curr_input[0, -1, 1].item()
            gravity = 0.0
            if current_rsi_norm > 0.8: gravity = -recent_volatility * 0.1
            elif current_rsi_norm < -0.8: gravity = recent_volatility * 0.1
            gravity *= abs(trend_suppression)
            vel_trend = (vel_trend * INERTIA_TREND) + ((target_trend - pos_trend) * FORCE_TREND) + gravity
            vel_trend = np.clip(vel_trend, -MAX_VELOCITY, MAX_VELOCITY)
            pos_trend += vel_trend
            local_vol_factor = np.std(s_diff.inverse_transform(start_win[:, 0].reshape(-1,1)))
            chaos = 1.0 + (abs(current_rsi_norm) * 2.0)
            stoch_noise_delta = (real_sigma_delta * chaos * torch.randn(1).item() * current_temp * 1.5)
            target_stoch = pos_trend + stoch_noise_delta
            tether_pull = (pos_trend - pos_stoch) * TETHER_STRENGTH
            vel_stoch = (vel_stoch * INERTIA_STOCH) + ((target_stoch - pos_stoch) * FORCE_STOCH) + tether_pull
            vel_stoch = np.clip(vel_stoch, -MAX_VELOCITY*3.0, MAX_VELOCITY*3.0)
            pos_stoch += vel_stoch
            if current_rsi_norm > 0.5: res_stiffness = 0.05
            else: res_stiffness = 0.15 
            if current_rsi_norm < -0.5: sup_stiffness = 0.05
            else: sup_stiffness = 0.15
            target_res = (pos_trend + raw_width_res * drift_res_factor)
            target_sup = (pos_trend - raw_width_sup * drift_sup_factor)
            vel_res = (vel_res * 0.85) + ((target_res - pos_res) * res_stiffness)
            pos_res += vel_res
            vel_sup = (vel_sup * 0.85) + ((target_sup - pos_sup) * sup_stiffness)
            pos_sup += vel_sup
            preds_trend.append(pos_trend); preds_stoch.append(pos_stoch)
            preds_res.append(pos_res); preds_sup.append(pos_sup)
            sim_history.append(pos_stoch)
            full_arr = np.array(sim_history)
            new_rsi_sc = s_rsi.transform([[calculate_rsi(full_arr, CONFIG["Rsi_Period"])[-1]]])[0][0]
            diff_arr = np.diff(full_arr)
            new_vol_sc = s_vol.transform([[np.std(diff_arr[-10:]) if len(diff_arr)>10 else 0]])[0][0]
            pct_chg = (full_arr[-1]-full_arr[-6])/(full_arr[-6]+1e-5)*100 if len(full_arr)>6 else 0
            new_roc_sc = s_roc.transform([[pct_chg]])[0][0]
            new_roc_sc = np.clip(new_roc_sc, -3.0, 3.0) 
            real_delta = vel_stoch 
            scaled_delta = s_diff.transform([[real_delta]])[0][0]
            scaled_delta = np.clip(scaled_delta, -3.0, 3.0)
            feat = torch.tensor([[[scaled_delta, new_rsi_sc, new_vol_sc, new_roc_sc]]], dtype=torch.float32).to(DEVICE)
            curr_input = torch.cat((curr_input[:, 1:, :], feat), dim=1)
    return preds_stoch, preds_trend, preds_res, preds_sup

def calculate_monte_carlo_probability(model, start_win, s_diff, s_rsi, s_vol, s_roc, last_price, history_prices):
    log_msg(f">>> [Probability] Running Monte Carlo ({CONFIG['Mc_Iterations']} runs)...")
    up_count = 0
    mc_temp = CONFIG["Temperature"] * CONFIG["Mc_Temp_Mult"]
    for _ in range(CONFIG["Mc_Iterations"]):
        p_stoch, _, _, _ = forecast(model, start_win, CONFIG["Prob_Lookahead_Mins"], s_diff, s_rsi, s_vol, s_roc, last_price, history_prices, temp_override=mc_temp, suppression=CONFIG["Mc_Trend_Suppression"])
        if p_stoch[-1] > last_price: up_count += 1
    probability_up = (up_count / CONFIG["Mc_Iterations"]) * 100.0
    return probability_up