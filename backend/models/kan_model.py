import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from backend.utils import CONFIG, DEVICE, log_msg, MODELS_DIR

# Klasy AdvancedKANLayer i TemporalKAN bez zmian...
class AdvancedKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.layernorm = nn.LayerNorm(input_dim)
        self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.spline_weight = nn.Parameter(torch.Tensor(output_dim, input_dim, grid_size))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
        self.base_activation = nn.SiLU()
    def forward(self, x):
        x = self.layernorm(x)
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        x_norm = torch.tanh(x) 
        spline_output = 0
        for i in range(self.grid_size):
            basis = torch.cos(math.pi * (i + 1) * x_norm)
            term = torch.matmul(basis, self.spline_weight[:, :, i].T)
            spline_output += term
        return base_output + spline_output

class TemporalKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.kan_head = AdvancedKANLayer(hidden_dim, 1, grid_size=6)
    def forward(self, x):
        _, hn = self.gru(x) 
        context_vector = hn[-1] 
        context_vector = self.dropout(context_vector)
        out = self.kan_head(context_vector)
        return torch.sigmoid(out)

def train_and_predict_kan(X, y, start_win):
    model_path = os.path.join(MODELS_DIR, "kan_model.pth")
    y_binary = (y > 0).float().unsqueeze(1).to(DEVICE)
    X = X.to(DEVICE) # Przesunięcie na GPU wewnątrz procesu

    kan_model = TemporalKAN(input_dim=4, hidden_dim=32).to(DEVICE)
    
    # Ładowanie
    if os.path.exists(model_path):
        try:
            kan_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        except: pass

    kan_opt = optim.AdamW(kan_model.parameters(), lr=CONFIG['Kan_Learning_Rate'], weight_decay=1e-4)
    kan_loss_fn = nn.BCELoss()
    kan_model.train()
    
    for ke in range(CONFIG['Kan_Epochs']): 
        kan_opt.zero_grad()
        out_kan = kan_model(X)
        loss_kan = kan_loss_fn(out_kan, y_binary)
        loss_kan.backward()
        kan_opt.step()

    # Zapis
    try:
        torch.save(kan_model.state_dict(), model_path)
    except: pass

    kan_model.eval()
    with torch.no_grad():
        kan_input = torch.tensor(start_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        raw_kan_prob = kan_model(kan_input).item() * 100.0
        kan_prob = 50.0 + ((raw_kan_prob - 50.0) * 0.25)
    
    log_msg(f">>> [4/4] T-KAN: {kan_prob:.2f}%")
    return kan_prob