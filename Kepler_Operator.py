import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- STEP 1: TEMPORAL ENCODING ---
class ContinuousTemporalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        # Define physical period limits in NORMALIZED YEARS
        min_period = 2.0 / 365.25
        max_period = 10.0
        
        omega_max = (2 * math.pi) / min_period
        omega_min = (2 * math.pi) / max_period
        
        exponent = torch.arange(0, d_model, 2).float() / d_model
        div_term = omega_max * torch.pow(omega_min / omega_max, exponent)
        
        self.register_buffer('div_term', div_term)

    def forward(self, t):
        t_expanded = t.unsqueeze(-1)  
        phases = t_expanded * self.div_term  
        
        pe = torch.zeros(t.size(0), t.size(1), self.d_model, device=t.device)
        pe[:, :, 0::2] = torch.sin(phases)
        pe[:, :, 1::2] = torch.cos(phases)
        return pe

# --- STEP 2: THE ENCODER & POOLING ---
class OrbitTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=5):
        super().__init__()
        self.d_model = d_model
        
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.time_encoding = ContinuousTemporalEncoding(d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.parameter_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 5)
        )
        
    def forward(self, seq_noisy):
        # CORRECTED SLICING: data_gen.py outputs [X, Y, Time]
        positions = seq_noisy[:, :, 0:2]
        times = seq_noisy[:, :, 2]
        
        spatial_feat = self.spatial_embedding(positions)
        temporal_feat = self.time_encoding(times)
        
        x = self.emb_norm(spatial_feat + temporal_feat)
        
        # No padding masks necessary since every orbit is exactly 64 points
        encoded = self.transformer(x)
        
        # Standard average pooling across the 64 valid points
        pooled = encoded.mean(dim=1)
        
        raw_parameters = self.parameter_head(pooled)
        return raw_parameters

# --- STEP 3: THE PHYSICS DECODER (Non-Dimensionalized) ---
class DifferentiableKepler(nn.Module):
    def __init__(self):
        super().__init__()
        self.G_norm = 132.0227

    def forward(self, a_norm, e, omega, M0, M_star_norm, t_norm):
        n = torch.sqrt(self.G_norm * M_star_norm / torch.clamp(a_norm**3, min=1e-4))
        
        M_t = M0 + n * t_norm
        
        E = M_t
        e_expanded = e.expand_as(E)
        
        for _ in range(8):
            denominator = torch.clamp(1.0 - e_expanded * torch.cos(E), min=0.05)
            E = E - (E - e_expanded * torch.sin(E) - M_t) / denominator
            
        x_orb = a_norm * (torch.cos(E) - e_expanded)
        y_orb = a_norm * torch.sqrt(torch.clamp(1 - e_expanded**2, min=1e-7)) * torch.sin(E)
        
        omega_exp = omega.expand_as(x_orb)
        x_physical = x_orb * torch.cos(omega_exp) - y_orb * torch.sin(omega_exp)
        y_physical = x_orb * torch.sin(omega_exp) + y_orb * torch.cos(omega_exp)
        
        return torch.stack([x_physical, y_physical], dim=-1)

class KeplerianOperator(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=5):
        super().__init__()
        self.transformer = OrbitTransformer(d_model, nhead, num_layers)
        self.kepler_solver = DifferentiableKepler()
        
    def forward(self, seq_noisy):
        raw_guess = self.transformer(seq_noisy)
        
        a_norm = 0.5 + F.softplus(raw_guess[:, 0:1]) * 3.0  
        e = torch.sigmoid(raw_guess[:, 1:2]) * 0.91 
        omega = raw_guess[:, 2:3] * torch.pi
        M0 = raw_guess[:, 3:4] * torch.pi
        M_norm = 1e-6 + F.softplus(raw_guess[:, 4:5]) * 2.0 
        
        # CORRECTED SLICING: Pass the 3rd column (Time) to the physics engine
        obs_times_norm = seq_noisy[:, :, 2] 
        
        normalized_positions = self.kepler_solver(a_norm, e, omega, M0, M_norm, obs_times_norm)
        
        return normalized_positions