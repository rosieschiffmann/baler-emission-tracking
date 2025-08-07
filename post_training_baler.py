import os
from types import SimpleNamespace

# Import the "manager" functions we need for the later steps
from baler.baler import (
    perform_compression,
    perform_decompression,
    perform_plotting,
)
from workspaces.CMS_workspace.CMS_project_v1.config.CMS_project_v1_config import set_config

def main():
    """
    This script continues the workflow using an ALREADY TRAINED model.
    It performs: Compress -> Decompress -> Plot.
    """
    
    # === 1. SETUP ===
    print("--- Creating Baler configuration to find existing results ---")
    config = SimpleNamespace()
    set_config(config)

    # These MUST match the settings from your training run
    config.epochs = 100
    config.model_type = "dense"

    # This path MUST point to the output from your previous successful training run
    project_name = f"baler_{config.epochs}_epochs"
    output_path = f"projects/{project_name}/output"
    
    # Define and create the remaining directories that these steps will need
    decompressed_output_path = os.path.join(output_path, "decompressed_output")
    plotting_path = os.path.join(output_path, "plotting")
    os.makedirs(decompressed_output_path, exist_ok=True)
    os.makedirs(plotting_path, exist_ok=True)
    
    print(f"--- Ready to continue workflow using results in: {output_path} ---\n")
    
    # === 2. PERFORM POST-TRAINING STEPS ===
    
    print("\n=== STARTING COMPRESSION ===")
    perform_compression(output_path=output_path, config=config, verbose=True)
    print("--- Compression Finished ---")

    print("\n=== STARTING DECOMPRESSION ===")
    perform_decompression(output_path=output_path, config=config, verbose=True)
    print("--- Decompression Finished ---")
    
    print("\n=== STARTING PLOTTING ===")
    perform_plotting(output_path=output_path, config=config, verbose=True)
    print(f"--- Plotting Finished. Plots saved in '{plotting_path}' ---")


if __name__ == "__main__":
    main()