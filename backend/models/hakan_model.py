import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# Zauważ, że importujemy teraz MODELS_DIR oraz log_msg
from backend.utils import DEVICE, CONFIG, MODELS_DIR, log_msg

class HahnBasis(nn.Module):
    def __init__(self, degree, alpha=0.0, beta=0.0):
        super().__init__()
        self.degree = degree
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, N_val=100):
        x_scaled = (x + 1) * (N_val - 1) / 2
        poly = [torch.ones_like(x_scaled)]
        if self.degree > 0:
            h1 = 1 - (x_scaled * (self.alpha + self.beta + 2)) / ((self.alpha + 1) * (N_val - 1))
            poly.append(h1)
        for n in range(1, self.degree):
            a_n = (2 * n + self.alpha + self.beta + 1) / (n + 1)
            b_n = (n + self.alpha + self.beta) / (n + 1)
            pn = (a_n * (x_scaled / N_val) * poly[-1] - b_n * poly[-2])
            poly.append(pn)
        return torch.stack(poly, dim=-1)

class HahnKANLayer(nn.Module):
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()
        self.basis = HahnBasis(degree)
        self.poly_weight = nn.Parameter(torch.randn(out_features, in_features, degree + 1) / (in_features ** 0.5))
        self.linear_shortcut = nn.Linear(in_features, out_features)

    def forward(self, x):
        res = self.linear_shortcut(x)
        x_poly = self.basis(x)
        if x.dim() == 3:
            poly_out = torch.einsum('bli d, oid -> blo', x_poly, self.poly_weight)
        else:
            poly_out = torch.einsum('bi d, oid -> bo', x_poly, self.poly_weight)
        return res + poly_out

class HaKANModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, pred_len, degree=3):
        super().__init__()
        self.layer1 = HahnKANLayer(input_dim, hidden_dim, degree)
        self.layer2 = HahnKANLayer(hidden_dim, pred_len, degree)

    def forward(self, x):
        x = x.view(x.size(0), -1) 
        x = self.layer1(x)
        x = torch.tanh(x) 
        x = self.layer2(x)
        return x


def train_hakan(model, X_train, y_train):
    # === DODANO: Ładowanie stanu z dysku ===
    model_path = os.path.join(MODELS_DIR, "hakan_model.pth")
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except Exception as e:
            log_msg(f"HaKAN: Nie udalo sie zaladowac starego modelu: {e}")
    # =======================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["HaKAN_Learning_Rate"], weight_decay=1e-4)
    criterion = nn.MSELoss()
    model.train()
    
    dataset_size = len(X_train)
    indices = np.arange(dataset_size)
    batch_size = CONFIG.get("HaKAN_Batch_Size", 64) 
    
    for _ in range(CONFIG["HaKAN_Epochs"]):
        np.random.shuffle(indices)
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_idx = indices[start_idx:end_idx]
            
            batch_X = X_train[batch_idx].to(DEVICE)
            batch_yh = y_train[batch_idx].to(DEVICE)
            
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_yh)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # === DODANO: Zapisywanie stanu na dysk po treningu ===
    try:
        torch.save(model.state_dict(), model_path)
    except Exception as e:
        log_msg(f"HaKAN: Nie udalo sie zapisac modelu: {e}")
    # =====================================================

def run_hakan_inference(model, start_win, steps, s_diff, last_price):
    model.eval()
    with torch.no_grad():
        input_ts = torch.tensor(start_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_norm = model(input_ts).cpu().numpy()[0]
        diffs = s_diff.inverse_transform(pred_norm.reshape(-1, 1)).flatten()  
        return last_price + np.cumsum(diffs)