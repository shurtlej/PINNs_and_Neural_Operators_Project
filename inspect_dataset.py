import h5py
import numpy as np
import os

def inspect_h5_file(filepath="orbits_data.h5"):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist in this directory.")
        return

    print(f"Opening '{filepath}'...")
    with h5py.File(filepath, 'r') as f:
        keys = list(f.keys())
        total_samples = len(keys)
        
        print("\n=== FILE SUMMARY ===")
        print(f"Total number of orbit samples (groups): {total_samples}")
        
        if total_samples == 0:
            print("Warning: The HDF5 file is empty!")
            return

        # Pick the first group to inspect the schema
        sample_key = keys[0]
        sample_group = f[sample_key]
        
        print(f"\n=== SAMPLE SCHEMA (First Group: '{sample_key}') ===")
        print("Datasets found inside this group:")
        for dataset_name in sample_group.keys():
            dataset = sample_group[dataset_name]
            # Print shape, dtype, and a preview of the min/max values to spot anomalies
            data = dataset[:]
            print(f"  - '{dataset_name}':")
            print(f"      Shape : {dataset.shape}")
            print(f"      Type  : {dataset.dtype}")
            print(f"      Range : Min = {np.min(data):.3e}, Max = {np.max(data):.3e}")
            if np.any(np.isnan(data)):
                print("      WARNING: Contains NaNs!")
        
        print("\nGroup Attributes (Keplerian parameters / Targets):")
        if len(sample_group.attrs) == 0:
            print("  - None")
        else:
            for attr_name, attr_val in sample_group.attrs.items():
                print(f"  - '{attr_name}': {attr_val}")

if __name__ == "__main__":
    inspect_h5_file()