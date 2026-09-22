import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import h5py
import numpy as np
import os
import warnings

# Use 'Agg' backend for matplotlib to prevent threading errors and headless server crashes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Suppress harmless PyTorch UserWarnings regarding nested tensors and norm_first
warnings.filterwarnings("ignore", category=UserWarning, message=".*nested_tensor.*")

# --- Import your newly overhauled Operator ---
from Kepler_Operator import KeplerianOperator

# --- Physical Scaling Constants ---
SCALE_T = 31536000.0        # 1 Year in seconds
SCALE_POS = 1e11            # ~1 AU in meters
SCALE_M = 1.989e30          # 1 Solar Mass

class OrbitDataset(Dataset):
    """
    Loads orbit groups from orbits_data.h5.
    Normalizes time and coordinates for stable network digestion,
    and extracts Keplerian attributes for physical metric monitoring.
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.keys = list(f.keys())
            
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            grp = f[self.keys[idx]]
            
            # Load raw coordinates and timestamps [64, 3] from generator
            # Statically sized at 64 observations, no padding.
            noisy_seq = grp['points_noisy'][:]  
            clean_seq = grp['points'][:]        
            
            # Extract underlying target parameters for diagnostics
            a = grp.attrs.get('a', 1.0 * SCALE_POS)
            e = grp.attrs.get('e', 0.0)
            omega = grp.attrs.get('omega', 0.0)
            M0 = grp.attrs.get('M0', 0.0)
            mass = grp.attrs.get('mass', 1.0 * SCALE_M)
            
        # ALIGNMENT: 
        # Columns are strictly [X, Y, Time]. 
        # Indices 0 and 1 are Spatial Coordinates -> scale by SCALE_POS.
        # Index 2 is Time -> scale by SCALE_T.
        noisy_norm = np.copy(noisy_seq)
        noisy_norm[:, 0:2] /= SCALE_POS
        noisy_norm[:, 2] /= SCALE_T
        
        clean_norm = np.copy(clean_seq)
        clean_norm[:, 0:2] /= SCALE_POS
        clean_norm[:, 2] /= SCALE_T
        
        physical_targets = np.array([a, e, omega, M0, mass], dtype=np.float32)
        
        return (
            torch.tensor(noisy_norm, dtype=torch.float32), 
            torch.tensor(clean_norm, dtype=torch.float32),
            torch.tensor(physical_targets, dtype=torch.float32)
        )

def train_kepler_operator(dataset_path="orbits_data.h5", epochs=150, batch_size=64, model_save_path="kepler_operator_best.pth"):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Database {dataset_path} not found. Ensure your generator ran successfully.")

    # Enforce strict reproducibility for train/val splits
    torch.manual_seed(42)
    np.random.seed(42)
    generator = torch.Generator().manual_seed(42)

    dataset = OrbitDataset(dataset_path)
    
    # 20% validation split to hold out a stable physical distribution
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Keplerian training loop on {device}...")
    print(f"Dataset Size: {len(dataset)} orbits | Train: {train_size} | Val: {val_size}")
    
    # Isolate a single fixed validation orbit to serve as our progress anchor
    anchor_noisy, anchor_clean, anchor_targets = val_dataset[0]
    anchor_noisy_batched = anchor_noisy.unsqueeze(0).to(device)
    
    # Initialize our Kepler Operator model
    model = KeplerianOperator(d_model=128, nhead=8, num_layers=5).to(device)
    
    # --- CHECKPOINT LOADER & LR TUNING ---
    starting_lr = 1e-3
    if os.path.exists(model_save_path):
        print(f"Loading existing model weights from {model_save_path}...")
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
            starting_lr = 1e-4 
            print("→ Success! Resuming training from last saved checkpoint.")
            print(f"→ Learning rate scaled down to {starting_lr} to protect pre-trained weights.")
        except Exception as e:
            print(f"→ Warning: Failed to load state dict: {e}")
            print("→ Starting training fresh.")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=starting_lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for noisy, clean, targets in train_loader:
            noisy, clean, targets = noisy.to(device), clean.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Extract raw predictions from the Transformer network
            raw_guess = model.transformer(noisy)
            
            # Map predictions to physical spaces using Kepler_Operator.py's leaky_relu limits
            pred_a_norm = 1.8 + torch.nn.functional.leaky_relu(raw_guess[:, 0:1], negative_slope=0.1)
            pred_a_norm = torch.clamp(pred_a_norm, min=0.3, max=4.0)
            
            pred_e_norm = 0.45 + torch.nn.functional.leaky_relu(raw_guess[:, 1:2], negative_slope=0.1)
            pred_e_norm = torch.clamp(pred_e_norm, min=0.0, max=0.91)
            
            pred_om_norm = raw_guess[:, 2:3] * torch.pi
            pred_m0_norm = raw_guess[:, 3:4] * torch.pi
            
            pred_M_norm = 0.25 + torch.nn.functional.leaky_relu(raw_guess[:, 4:5], negative_slope=0.1)
            pred_M_norm = torch.clamp(pred_M_norm, min=1e-6, max=1.0)
            
            # Normalize target parameters
            true_a_norm = targets[:, 0:1] / SCALE_POS
            true_e = targets[:, 1:2]
            true_om = targets[:, 2:3]
            true_m0 = targets[:, 3:4]
            true_M_norm = targets[:, 4:5] / SCALE_M
            
            # --- PURE PARAMETER-SPACE LOSS ---
            # Standard parameters are calculated linearly.
            # Angles are projected through sine and cosine to treat degenerate wrap limits as identical.
            loss = (
                criterion(pred_a_norm, true_a_norm) +
                criterion(pred_e_norm, true_e) +
                criterion(torch.sin(pred_om_norm), torch.sin(true_om)) +
                criterion(torch.cos(pred_om_norm), torch.cos(true_om)) +
                criterion(torch.sin(pred_m0_norm), torch.sin(true_m0)) +
                criterion(torch.cos(pred_m0_norm), torch.cos(true_m0)) +
                criterion(pred_M_norm, true_M_norm)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        
        a_errors = []
        e_errors = []
        m_errors = []
        
        with torch.no_grad():
            for noisy, clean, targets in val_loader:
                noisy, clean, targets = noisy.to(device), clean.to(device), targets.to(device)
                
                # Evaluate standard coordinate trajectory MSE on validation set for performance tracking
                pred_positions = model(noisy)
                true_positions = clean[:, :, 0:2]
                
                loss_coord = criterion(pred_positions, true_positions)
                val_loss += loss_coord.item()
                
                # --- Diagnostic Parameter Extraction ---
                raw_guess = model.transformer(noisy)
                
                pred_a_norm = 1.8 + torch.nn.functional.leaky_relu(raw_guess[:, 0], negative_slope=0.1)
                pred_a_norm = torch.clamp(pred_a_norm, min=0.3, max=4.0)
                
                pred_e_norm = 0.45 + torch.nn.functional.leaky_relu(raw_guess[:, 1], negative_slope=0.1)
                pred_e_norm = torch.clamp(pred_e_norm, min=0.0, max=0.91)
                
                pred_M_norm = 0.25 + torch.nn.functional.leaky_relu(raw_guess[:, 4], negative_slope=0.1)
                pred_M_norm = torch.clamp(pred_M_norm, min=1e-6, max=1.0)
                
                pred_a_m = (pred_a_norm * SCALE_POS).cpu().numpy()
                pred_e_np = pred_e_norm.cpu().numpy()
                pred_M_kg = (pred_M_norm * SCALE_M).cpu().numpy()
                
                true_a = targets[:, 0].cpu().numpy()
                true_e = targets[:, 1].cpu().numpy()
                true_M = targets[:, 4].cpu().numpy()
                
                a_errors.extend(np.abs(pred_a_m - true_a) / 1e3)  # km
                e_errors.extend(np.abs(pred_e_np - true_e))
                m_errors.extend(np.abs(pred_M_kg - true_M) / SCALE_M) # M_sun
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # --- Generate Dynamic Non-Blocking Progress Plot ---
        with torch.no_grad():
            pred_anchor_pos = model(anchor_noisy_batched).squeeze(0).cpu().numpy()
            
            raw_anchor_guess = model.transformer(anchor_noisy_batched)
            
            p_a = 1.8 + torch.nn.functional.leaky_relu(raw_anchor_guess[0, 0], negative_slope=0.1)
            p_a = torch.clamp(p_a, min=0.3, max=4.0)
            
            p_e = 0.45 + torch.nn.functional.leaky_relu(raw_anchor_guess[0, 1], negative_slope=0.1)
            p_e = torch.clamp(p_e, min=0.0, max=0.91)
            
            p_M = 0.25 + torch.nn.functional.leaky_relu(raw_anchor_guess[0, 4], negative_slope=0.1)
            p_M = torch.clamp(p_M, min=1e-6, max=1.0)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(0, 0, 'y*', markersize=18, label="Central Mass")
        
        # Target coordinates are columns 0 (X) and 1 (Y)
        ax.plot(anchor_clean[:, 0], anchor_clean[:, 1], 'g-', label="True Trajectory", linewidth=2.5, alpha=0.8)
        
        # Noisy observations are columns 0 (X) and 1 (Y)
        ax.scatter(anchor_noisy[:, 0], anchor_noisy[:, 1], c='blue', s=20, label="Noisy Observations", alpha=0.5, zorder=3)
        
        # Network predictions are pure positions [x, y]
        ax.plot(pred_anchor_pos[:, 0], pred_anchor_pos[:, 1], 'r--', label="Predicted Orbit (Neural Operator)", linewidth=2.5, zorder=4)
        
        ax.set_title(f"Epoch {epoch+1:03d} Trajectory Fit [Pure Parametric Loss]\n"
                     f"True a: {anchor_targets[0]/SCALE_POS:.2f} AU | Pred a: {p_a.item():.2f} AU\n"
                     f"True e: {anchor_targets[1]:.2f} | Pred e: {p_e.item():.2f}\n"
                     f"True Mass: {anchor_targets[4]/SCALE_M:.3f} M_sun | Pred Mass: {p_M.item():.3f} M_sun")
        ax.set_xlabel("X Position (Normalized)")
        ax.set_ylabel("Y Position (Normalized)")
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig("epoch_progress.png", dpi=150)
        plt.close(fig) # Prevent memory leaks
        
        # Save checkpoint if spatial trajectory error improves
        saved_status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            saved_status = "[Checkpoint Saved]"
            
        if (epoch + 1) % 5 == 0 or saved_status:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:03d}/{epochs} | LR: {current_lr:.1e} | Train Param Loss: {train_loss:.6f} | Val Traj MSE: {val_loss:.6f} {saved_status}")
            print(f"    ↳ Mean Diagnostics | Δa: {np.mean(a_errors):,.0f} km | Δe: {np.mean(e_errors):.3f} | ΔMass: {np.mean(m_errors):.3f} M_sun")

if __name__ == "__main__":
    train_kepler_operator()