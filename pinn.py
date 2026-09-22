import torch
import torch.nn as nn
import h5py
import numpy as np
import os
import math

# Use 'Agg' backend for headless saving without GUI thread blocking
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Physical Scaling Constants ---
SCALE_T = 31536000.0        # 1 Year in seconds
SCALE_POS = 1e11            # ~1 AU in meters
SCALE_M = 1.989e30          # 1 Solar Mass

# --- THE UPGRADED TIME-WARPED PINN WITH STRUCTURAL LOW-PASS FILTERING ---
class KeplerPINN(nn.Module):
    """
    Coordinate MLP mapping time (t) to positions (x,y).
    Uses a Fourier Feature mapping layer restricted to very low frequencies
    to act as a hardware-level low-pass filter, preventing high-frequency wrinkling.
    """
    def __init__(self, hidden_dim=128, num_frequencies=4):
        super().__init__()
        
        # Time-Warping Subnet: Maps linear time (t) to eccentric time (tau).
        self.time_warp = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # --- LOW-FREQUENCY SPECTRAL ANCHORING ---
        # We structurally limit the frequencies to only the fundamental orbital periods (0.5 to 5.0 years)
        # This completely bars the network from representing sharp kinks or high-frequency wiggles.
        periods = np.logspace(np.log10(5.0), np.log10(0.5), num_frequencies)
        frequencies = 2.0 * np.pi / periods
        self.register_buffer('frequencies', torch.tensor(frequencies, dtype=torch.float32))
        
        in_dim = 2 * num_frequencies
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),  # Mandatory for continuous second derivatives
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # Outputs: [x, y]
        )

    def forward(self, t):
        # Learn a warped timeline.
        tau = t + self.time_warp(t)
        
        # Project warped time into Fourier frequencies
        phases = tau * self.frequencies.unsqueeze(0)
        features = torch.cat([torch.sin(phases), torch.cos(phases)], dim=-1)
        return self.net(features)


# --- LOAD DATA ENGINE ---
def load_single_orbit(h5_path="orbits_data.h5"):
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Could not find {h5_path}. Run your dataset generator first.")
        
    with h5py.File(h5_path, 'r') as f:
        first_key = list(f.keys())[0]
        grp = f[first_key]
        noisy_seq = grp['points_noisy'][:]  
        clean_seq = grp['points'][:]        
        true_mass = grp.attrs.get('mass', 1.0 * SCALE_M)
        
    t_norm = noisy_seq[:, 2] / SCALE_T
    x_noisy_norm = noisy_seq[:, 0] / SCALE_POS
    y_noisy_norm = noisy_seq[:, 1] / SCALE_POS
    x_clean_norm = clean_seq[:, 0] / SCALE_POS
    y_clean_norm = clean_seq[:, 1] / SCALE_POS
    
    t_tensor = torch.tensor(t_norm, dtype=torch.float32).unsqueeze(-1)
    coords_noisy = torch.tensor(np.column_stack((x_noisy_norm, y_noisy_norm)), dtype=torch.float32)
    coords_clean = torch.tensor(np.column_stack((x_clean_norm, y_clean_norm)), dtype=torch.float32)
    
    return t_tensor, coords_noisy, coords_clean, true_mass


# --- RUN RUNTIME CORE ---
def train_pinn(epochs=4000, lr=1e-3):
    t_obs, pos_obs, pos_clean, true_mass = load_single_orbit()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_obs = t_obs.to(device)
    pos_obs = pos_obs.to(device)
    
    # Restricting frequencies to 4 structural channels
    pinn = KeplerPINN(hidden_dim=128, num_frequencies=4).to(device)
    
    # Stellar Bias Initialization: Center model predictions on observation track
    with torch.no_grad():
        mean_pos = pos_obs.mean(dim=0)
        pinn.net[-1].bias.copy_(mean_pos)
        print(f"PINN coordinates initialized far from origin at center: {mean_pos.cpu().numpy()}")
        
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    
    G_norm = 132.0227
    mu = G_norm * (true_mass / SCALE_M)
    
    print(f"Running Fixed Coordinate-MLP PINN Loop on {device}...")
    
    weight_physics = 0

    for epoch in range(epochs + 1):
        pinn.train()
        optimizer.zero_grad()
        
        # Temporal Curriculum Windowing
        if epoch < 1000:
            active_points = 20
        elif epoch < 2000:
            active_points = 42
        else:
            active_points = 64
            
        pred_obs = pinn(t_obs[:active_points])
        loss_data = torch.mean((pred_obs - pos_obs[:active_points]) ** 2)
        
        # --- PHYSICS LAYER CONSTRAINTS WITH ANNEALED SOFTENING ---
        t_physics = torch.linspace(t_obs.min().item(), t_obs.max().item(), 200, device=device).unsqueeze(-1)
        t_physics.requires_grad_(True)
        
        # --- MONOTONICITY GUARD ---
        tau = t_physics + pinn.time_warp(t_physics)
        dtaudt = torch.autograd.grad(tau, t_physics, torch.ones_like(tau), create_graph=True)[0]
        loss_monotonicity = torch.mean(torch.clamp(0.1 - dtaudt, min=0.0) ** 2)
        
        pred_phys = pinn(t_physics)
        x_pred = pred_phys[:, 0:1]
        y_pred = pred_phys[:, 1:2]
        
        # Exact Autograd Graph Derivation
        dxdt = torch.autograd.grad(x_pred, t_physics, torch.ones_like(x_pred), create_graph=True)[0]
        dydt = torch.autograd.grad(y_pred, t_physics, torch.ones_like(y_pred), create_graph=True)[0]
        d2xdt2 = torch.autograd.grad(dxdt, t_physics, torch.ones_like(dxdt), create_graph=True)[0]
        d2ydt2 = torch.autograd.grad(dydt, t_physics, torch.ones_like(dydt), create_graph=True)[0]
        
        # Softening decays to prevent force explosion
        if epoch < 2000:
            softening = 0.05 * (1.0 - (epoch / 2000.0)) + 1e-5
        else:
            softening = 1e-5
            
        r = torch.sqrt(x_pred**2 + y_pred**2 + softening)
        
        # Newtonian Gravitational Loss
        residual_x = d2xdt2 + (mu * x_pred) / (r**3)
        residual_y = d2ydt2 + (mu * y_pred) / (r**3)
        loss_gravity = torch.mean(residual_x**2) + torch.mean(residual_y**2)
        
        # --- ANGULAR MOMENTUM REGULARIZATION ---
        L = x_pred * dydt - y_pred * dxdt
        loss_angular_momentum = torch.var(L) + torch.mean(torch.clamp(0.01 - L, min=0.0) ** 2)
        
        # Aggregate physical losses
        loss_physics = loss_gravity + 10.0 * loss_angular_momentum + 5.0 * loss_monotonicity
        
        # --- Dynamic Graph Render Output ---
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | data_mse: {loss_data.item():.5f} | phys_weight: {weight_physics:.5e} | L_var: {torch.var(L).item():.3e}")
            
            with torch.no_grad():
                t_plot = torch.linspace(t_obs.min().item(), t_obs.max().item(), 300, device=device).unsqueeze(-1)
                dense_pred = pinn(t_plot).cpu().numpy()
                
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(0, 0, 'y*', markersize=15, label="Central Star")
            ax.plot(pos_clean[:, 0], pos_clean[:, 1], 'g-', label="Ground Truth", linewidth=2.5, alpha=0.7)
            ax.scatter(pos_obs[:active_points, 0].cpu().numpy(), pos_obs[:active_points, 1].cpu().numpy(), c='blue', s=15, label="Active Telemetry", alpha=0.6)
            ax.plot(dense_pred[:, 0], dense_pred[:, 1], 'r--', label="Cyclic Fourier PINN Curve", linewidth=2.5)
            
            ax.set_title(f"Cyclic Kepler PINN Loop | Epoch {epoch:04d}\nActive Points: {active_points}")
            ax.set_xlabel("Normalized X")
            ax.set_ylabel("Normalized Y")
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            
            plt.tight_layout()
            plt.savefig("pinn_orbit_fit.png", dpi=150)
            plt.close(fig)
        
        # --- THE SOLUTION: CYCLIC PHYSICS SCHEDULER ---
        # Epochs 0 - 1000: Pure Spatial Warmup to construct the basic trajectory
        if epoch % 100 == 0:
            weight_physics = float(input("weight:"))
                
        loss = loss_data + weight_physics * loss_physics
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
        optimizer.step()

if __name__ == "__main__":
    train_pinn()