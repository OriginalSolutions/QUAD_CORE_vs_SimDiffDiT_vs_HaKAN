import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from backend.utils import CONFIG, DEVICE, log_msg, MODELS_DIR

# ==============================================================================
# ARCHITEKTURA SIECI (3 Wyjścia: Mu, Sigma, Quantiles)
# ==============================================================================

class DualModeNetwork(nn.Module):
    def __init__(self, input_dim=4, d_model=64, n_layers=2):
        super().__init__()
        # Główna warstwa rekurencyjna (LSTM)
        self.lstm = nn.LSTM(input_dim, d_model, num_layers=n_layers, batch_first=True, dropout=0.2)
        
        # Warstwa pośrednia
        self.shared_layer = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU()
        )
        
        # --- GŁOWICE (HEADS) ---
        
        # 1. Główna prognoza (Mean/Trend)
        self.mu_head = nn.Linear(32, 1)
        
        # 2. Zmienność (Sigma) - musi być dodatnia (Softplus)
        self.sigma_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # 3. Kwantyle (Szerokość kanałów) - zwraca 2 wartości, muszą być dodatnie
        self.quantile_head = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softplus()
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        out, _ = self.lstm(x)
        
        # Bierzemy ostatni krok czasowy
        last_step = out[:, -1, :]
        
        # Wspólne cechy
        shared_features = self.shared_layer(last_step)
        
        # Wyliczenie trzech składowych
        mu = self.mu_head(shared_features)
        sigma = self.sigma_head(shared_features)
        quantiles = self.quantile_head(shared_features)
        
        # ZWRACAMY 3 WARTOŚCI (zgodnie z wymaganiem monte_carlo.py)
        return mu, sigma, quantiles

# ==============================================================================
# FUNKCJA TRENUJĄCA Z ZAPISEM STANU (PERSISTENCE)
# ==============================================================================

def train_dual_mode_model(model, X, y):
    """
    Ładuje wagi -> Trenuje (mu, sigma, quantiles) -> Zapisuje wagi
    """
    model_path = os.path.join(MODELS_DIR, "neural_trend_model.pth")
    
    # 1. ŁADOWANIE STANU
    if os.path.exists(model_path):
        try:
            saved_state = torch.load(model_path, map_location=DEVICE)
            model.load_state_dict(saved_state)
            # log_msg("[NeuralTrend] Loaded weights from disk.")
        except Exception as e:
            # Jeśli architektura się zmieniła (np. dodaliśmy nowe głowice), 
            # stare wagi mogą nie pasować. Wtedy trenujemy od nowa.
            log_msg(f"!!! [NeuralTrend] Weight mismatch (architecture changed?): {e}. Starting fresh.")
    
    # Przeniesienie na urządzenie (jeśli X, y są na CPU, a model na GPU lub odwrotnie)
    # W multiprocessing device jest przekazywane dynamicznie, ale dla pewności:
    device = next(model.parameters()).device
    X = X.to(device)
    y = y.to(device) # y shape: [batch] lub [batch, 1]
    
    # Upewniamy się, że y ma odpowiedni kształt
    if len(y.shape) == 1:
        y = y.unsqueeze(1)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    mse_criterion = nn.MSELoss()
    
    model.train()
    
    # Krótki cykl dotrenowywania (np. 50 epok)
    epochs = 50 
    
    for _ in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (teraz zwraca 3 wartości)
        mu, sigma, quantiles = model(X)
        
        # --- LOSS FUNCTION ---
        # 1. Główny cel: mu ma przewidywać y (MSE)
        loss_mu = mse_criterion(mu, y)
        
        # 2. Cel pomocniczy: Sigma powinna odzwierciedlać błąd predykcji.
        #    Uczymy sigmę, by dążyła do wartości błędu bezwzględnego (Residuals).
        #    Detach() jest ważne, żeby sigma nie "psuła" mu (nie przesuwała średniej, żeby dopasować sigmę).
        residuals = torch.abs(mu.detach() - y)
        loss_sigma = mse_criterion(sigma, residuals)
        
        # 3. Cel pomocniczy: Kwantyle też powinny reagować na zmienność.
        #    Dla uproszczenia uczymy je również na podstawie residuali.
        loss_quant = mse_criterion(quantiles, residuals.repeat(1, 2))
        
        # Suma strat
        total_loss = loss_mu + (loss_sigma * 0.5) + (loss_quant * 0.5)
        
        total_loss.backward()
        optimizer.step()
        
    # 2. ZAPISYWANIE STANU
    try:
        torch.save(model.state_dict(), model_path)
        # log_msg("[NeuralTrend] Saved weights.")
    except Exception as e:
        log_msg(f"!!! [NeuralTrend] Save failed: {e}")