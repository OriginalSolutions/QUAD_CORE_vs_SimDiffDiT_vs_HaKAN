import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import os
from backend.utils import CONFIG, DEVICE, log_msg, MODELS_DIR

# ==============================================================================
# ARCHITEKTURA DiT (Bez zmian)
# ==============================================================================
# ... (Zostaw klasy AdaLN, DiTTransformerBlock, SimDiffHorizonModel bez zmian) ...
# Wklejam je dla kompletności, ale możesz zostawić te co masz, jeśli są identyczne.
class AdaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.embedding = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2))
    def forward(self, x, t_emb):
        gamma, beta = self.embedding(t_emb).chunk(2, dim=-1)
        return self.norm(x) * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

class DiTTransformerBlock(nn.Module):
    def __init__(self, hidden_size, nhead, dropout=0.1):
        super().__init__()
        self.ada_norm1 = AdaLN(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)
        self.ada_norm2 = AdaLN(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size * 4, hidden_size))
    def forward(self, x, t_emb):
        norm_x = self.ada_norm1(x, t_emb)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        norm_x = self.ada_norm2(x, t_emb)
        x = x + self.mlp(norm_x)
        return x

class SimDiffHorizonModel(nn.Module):
    def __init__(self, input_dim=4, hidden_size=128, depth=4, nhead=4, hist_len=10, pred_len=5):
        super().__init__()
        self.pred_len = pred_len
        self.hist_embed = nn.Linear(input_dim, hidden_size)
        self.hist_pos = nn.Parameter(torch.randn(1, hist_len, hidden_size))
        self.future_embed = nn.Linear(1, hidden_size)
        self.future_pos = nn.Parameter(torch.randn(1, pred_len, hidden_size))
        self.t_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([DiTTransformerBlock(hidden_size, nhead) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, 1)
    def get_timestep_embedding(self, timesteps, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
        args = timesteps[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    def forward(self, history, x_t, t):
        h = self.hist_embed(history) + self.hist_pos
        f = self.future_embed(x_t) + self.future_pos
        x = torch.cat([h, f], dim=1)
        t_emb = self.get_timestep_embedding(t, x.shape[-1])
        t_emb = self.t_mlp(t_emb)
        for block in self.blocks: x = block(x, t_emb)
        x = self.final_norm(x)
        future_out = x[:, -self.pred_len:, :]
        return self.head(future_out)

# ==============================================================================
# TRENING Z PERSISTENCE (Zapisywanie Stanu)
# ==============================================================================

def train_simdiff(model, X_train, y_train_vector):
    # Ścieżka do pliku z wagami
    model_path = os.path.join(MODELS_DIR, "simdiffdit_model.pth")
    
    # 1. Ładowanie wag (Dotrenowywanie)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            # log_msg("[SimDiff] Loaded weights from disk.")
        except Exception as e:
            log_msg(f"!!! [SimDiff] Load failed: {e}. Training from scratch.")
    else:
        log_msg("[SimDiff] No saved weights. Starting fresh.")

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["SimDiff_Learning_Rate"])
    loss_fn = nn.MSELoss()
    T_steps = CONFIG["SimDiff_T_Steps"]
    
    # Przesunięcie na GPU (bo prepare_data zwraca CPU)
    X_train = X_train.to(DEVICE)
    y_train_vector = y_train_vector.to(DEVICE)
    x_0 = y_train_vector.unsqueeze(-1)
    
    model.train()
    
    # Linear Schedule
    betas = torch.linspace(0.0001, 0.02, T_steps, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # log_msg(f">>> [SimDiff] Training ({CONFIG['SimDiff_Epochs']} ep)...")
    
    for epoch in range(CONFIG["SimDiff_Epochs"]):
        optimizer.zero_grad()
        B = X_train.shape[0]
        t = torch.randint(0, T_steps, (B,), device=DEVICE).long()
        ac_t = alphas_cumprod[t].reshape(B, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(ac_t) * x_0 + torch.sqrt(1 - ac_t) * noise
        
        pred_x0 = model(X_train, x_t, t.float())
        loss = loss_fn(pred_x0, x_0)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Opcjonalnie: log co 50 epok żeby nie śmiecić w konsoli
        if (epoch+1) % 50 == 0:
             print(f"    [SimDiff] Ep {epoch+1} | Loss: {loss.item():.4f}", flush=True)

    # 2. Zapisywanie stanu po treningu
    try:
        torch.save(model.state_dict(), model_path)
        # log_msg("[SimDiff] Saved weights to disk.")
    except Exception as e:
        log_msg(f"!!! [SimDiff] Save failed: {e}")

def run_simdiff_inference(model, start_win, steps, s_diff, s_rsi, s_vol, s_roc, last_price):
    M = CONFIG["SimDiff_M"]
    K = CONFIG["SimDiff_K"]
    T = CONFIG["SimDiff_T_Steps"]
    
    hist_input = torch.tensor(start_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    hist_input = hist_input.repeat(M, 1, 1)
    
    x_t = torch.randn(M, steps, 1, device=DEVICE)
    
    betas = torch.linspace(0.0001, 0.02, T, device=DEVICE)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    model.eval()
    with torch.no_grad():
        for i in reversed(range(T)):
            t = torch.full((M,), i, device=DEVICE).float()
            at = alphas_cumprod[i]
            at_prev = alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0).to(DEVICE)
            pred_x0 = model(hist_input, x_t, t)
            pred_x0 = torch.clamp(pred_x0, -4.0, 4.0)
            
            if i > 0:
                noise = torch.randn_like(x_t)
                denominator = torch.sqrt(1 - at)
                if denominator < 1e-6: denominator = 1e-6
                pred_noise = (x_t - torch.sqrt(at) * pred_x0) / denominator
                sigma = torch.sqrt(betas[i])
                dir_term = 1 - at_prev - sigma**2
                if dir_term < 0: dir_term = torch.tensor(0.0).to(DEVICE)
                x_t = torch.sqrt(at_prev) * pred_x0 + torch.sqrt(dir_term) * pred_noise + sigma * noise
            else:
                x_t = pred_x0

    raw_results = x_t.squeeze(-1).cpu().numpy()
    if np.isnan(raw_results).any():
        raw_results = np.nan_to_num(raw_results, nan=0.0)
    
    samples_per_group = M // K
    if samples_per_group < 1: samples_per_group = 1
    
    group_means = []
    for k in range(K):
        start = k * samples_per_group
        end = min((k + 1) * samples_per_group, M)
        if start >= M: break
        group_batch = raw_results[start:end, :] 
        group_means.append(np.mean(group_batch, axis=0))
    
    if not group_means: final_norm_diffs = np.zeros(steps)
    else:
        means_matrix = np.vstack(group_means)
        final_norm_diffs = np.median(means_matrix, axis=0)
        final_norm_diffs = final_norm_diffs * 2.0
    
    preds = []
    curr_price = last_price
    for d_norm in final_norm_diffs:
        try: real_diff = s_diff.inverse_transform([[d_norm]])[0][0]
        except: real_diff = 0.0
        curr_price += real_diff
        preds.append(curr_price)
    return preds   