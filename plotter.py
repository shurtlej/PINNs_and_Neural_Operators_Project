import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import sys

def find_dataset():
    """Identifies which of the two potential HDF5 files is present in the workspace."""
    candidates = ["orbits_data.h5", "orbital_training_data.h5"]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def plot_orbit(filepath, mode="random"):
    """
    Loads and plots an orbit from the specified H5 file.
    
    Parameters:
        filepath (str): Path to the HDF5 file.
        mode (str): 'random' to select a random orbit,
                    'max_e' to select the orbit with the highest eccentricity,
                    'min_e' to select the orbit with the lowest eccentricity.
    """
    print(f"Opening dataset: {filepath}")
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        if not keys:
            print("Error: The HDF5 file is empty!")
            return
        
        # --- SELECT THE TARGET ORBIT ---
        selected_key = None
        
        if mode == "random" or len(keys) < 2:
            selected_key = random.choice(keys)
            print(f"Selected random sample: '{selected_key}'")
        else:
            # We want to scan the attributes to find the extreme cases
            print(f"Scanning {len(keys)} orbits for the '{mode}' outlier...")
            best_val = -1.0 if mode == "max_e" else float('inf')
            
            for k in keys:
                grp = f[k]
                # Handle different potential naming conventions for eccentricity
                e_val = None
                if 'e' in grp.attrs:
                    e_val = grp.attrs['e']
                elif 'eccentricity' in grp.attrs:
                    e_val = grp.attrs['eccentricity']
                elif 'target_M' in grp.attrs:
                    # In train_operator.py style, target initial states are stored.
                    # We approximate eccentricity sorting or fallback to random
                    pass
                
                if e_val is not None:
                    if mode == "max_e" and e_val > best_val:
                        best_val = e_val
                        selected_key = k
                    elif mode == "min_e" and e_val < best_val:
                        best_val = e_val
                        selected_key = k
            
            if selected_key is None:
                selected_key = random.choice(keys)
                print(f"Could not find eccentricity attributes. Falling back to random: '{selected_key}'")
            else:
                print(f"Found outlier '{selected_key}' with eccentricity: {best_val:.4f}")

        # --- EXTRACT AND PROCESS DATA ---
        grp = f[selected_key]
        
        # Schema Detection
        # Schema A (orbits_data.h5): 'points' and 'points_noisy'
        if 'points' in grp and 'points_noisy' in grp:
            clean_data = grp['points'][:]        # Shape: [N, 3] -> [X, Y, Time]
            noisy_data = grp['points_noisy'][:]  # Shape: [N, 3] -> [X, Y, Time]
            
            # extract clean coordinates
            x_clean, y_clean, t_clean = clean_data[:, 0], clean_data[:, 1], clean_data[:, 2]
            # extract noisy coordinates
            x_noisy, y_noisy, t_noisy = noisy_data[:, 0], noisy_data[:, 1], noisy_data[:, 2]
            
        # Schema B (orbital_training_data.h5 / save_orbit_h5.py): 'sequence'/'sequence_noisy', 'attention_mask'
        elif 'sequence' in grp or 'sequence_noisy' in grp or 'clean_sequence' in grp:
            seq_key = 'sequence' if 'sequence' in grp else 'sequence_noisy'
            noisy_data = grp[seq_key][:]
            clean_data = grp['clean_sequence'][:] if 'clean_sequence' in grp else grp['sequence_clean'][:]
            mask = grp['attention_mask'][:]
            
            # Filter out padding elements using the attention mask
            valid_idx = (mask == 1.0)
            noisy_valid = noisy_data[valid_idx]
            clean_valid = clean_data[valid_idx]
            
            # File structure is [Time, X, Y] for Schema B
            t_noisy, x_noisy, y_noisy = noisy_valid[:, 0], noisy_valid[:, 1], noisy_valid[:, 2]
            t_clean, x_clean, y_clean = clean_valid[:, 0], clean_valid[:, 1], clean_valid[:, 2]
        else:
            print("Error: Could not identify the dataset schema. Keys present:", list(grp.keys()))
            return

        # --- METADATA EXTRACTION ---
        title_metadata = []
        for attr_key, attr_val in grp.attrs.items():
            # Format large numbers nicely
            if isinstance(attr_val, (int, float)):
                if attr_val > 1e9:
                    title_metadata.append(f"{attr_key}: {attr_val:.2e}")
                else:
                    title_metadata.append(f"{attr_key}: {attr_val:.3f}")
            else:
                title_metadata.append(f"{attr_key}: {attr_val}")
        metadata_str = " | ".join(title_metadata)

        # --- PLOTTING ---
        fig, ax = plt.subplots(figsize=(9, 8))
        
        # Plot Central Mass
        ax.plot(0, 0, 'y*', markersize=18, label="Central Star", zorder=6)
        
        # Plot continuous uncorrupted path
        ax.plot(x_clean, y_clean, 'g-', label="Ground Truth Trajectory", linewidth=2.5, alpha=0.8, zorder=3)
        
        # Plot noisy observations with time color-coding
        scatter = ax.scatter(x_noisy, y_noisy, c=t_noisy, cmap='viridis', s=35, 
                             edgecolors='black', linewidths=0.5, label="Noisy Telemetry Points", zorder=4)
        
        # Format axes
        ax.set_title(f"Visualizing Orbit: {selected_key}\n{metadata_str}", fontsize=11, pad=10)
        ax.set_xlabel("X Position (Normalized or Meters)")
        ax.set_ylabel("Y Position (Normalized or Meters)")
        ax.axis('equal')  # Critical to preserve the geometric eccentricity of the ellipse
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')
        
        # Add colorbar for time progression
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label("Observation Time (Normalized or Seconds)")
        
        plt.tight_layout()
        
        # Save output image
        output_img = "random_orbit_plot.png"
        plt.savefig(output_img, dpi=150)
        print(f"Plot saved successfully to '{output_img}'!")
        plt.close(fig)

if __name__ == "__main__":
    db_file = find_dataset()
    if db_file is None:
        print("Error: Could not find any orbit database file (e.g., 'orbits_data.h5' or 'orbital_training_data.h5').")
        print("Please ensure your dataset generation script has been run in this directory.")
        sys.exit(1)
        
    # Options: "random", "max_e", "min_e"
    # Change "random" here to "max_e" to view the most eccentric orbit in your dataset!
    mode_selection = "random" 
    
    if len(sys.argv) > 1 and sys.argv[1] in ["random", "max_e", "min_e"]:
        mode_selection = sys.argv[1]
        
    plot_orbit(db_file, mode=mode_selection)