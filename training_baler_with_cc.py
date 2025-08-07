import os
from types import SimpleNamespace
from codecarbon import EmissionsTracker

# === STEP 1: Import the high-level "manager" function and your config ===
from baler.baler import perform_training
from workspaces.CMS_workspace.CMS_project_v1.config.CMS_project_v1_config import set_config


def main():
    """
    Main function to run Baler training while tracking its emissions.
    """
    
    # === STEP 2: SETUP (UNTRACKED) ===
    # We only need to create the config object and define an output path.
    
    print("--- Creating and populating Baler configuration ---")
    config = SimpleNamespace()
    set_config(config)

    config.epochs = 1000  # Override the default value of 25
    config.model_type = "dense" 

    # Create a unique directory name for this experiment based on the epoch count
    project_name = f"baler_{config.epochs}_epochs"
    output_path = f"projects/{project_name}/output"

    # The rest of the paths are built from this, so they will be unique too
    training_path = os.path.join(output_path, "training")
    compressed_output_path = os.path.join(output_path, "compressed_output")

    # Create all necessary directories
    os.makedirs(training_path, exist_ok=True)
    os.makedirs(compressed_output_path, exist_ok=True)
    
    print(f"--- Configuration ready. Output will be saved to: {output_path} ---\n")

    
    # === STEP 3: TRAINING (TRACKED) ===
    # We set up the tracker to measure only the perform_training function call.
    
    output_dir = "codecarbon_logs"
    os.makedirs(output_dir, exist_ok=True)
    
    tracker = EmissionsTracker(
    project_name=f"baler_training_{config.epochs}_epochs",
    output_dir=output_dir,
    output_file=f"emissions_{config.epochs}_epochs.csv",
    log_level="info")

    try:
        tracker.start()
        print("--- CodeCarbon tracker started. Starting Baler Training... ---")
        
        # === THE CORRECT, SIMPLE CALL ===
        # We call the high-level "manager" function. It does all the work internally.
        # We pass verbose=True to see all of Baler's normal print statements.
        perform_training(output_path=output_path, config=config, verbose=True)
        
        print(f"--- Baler Training Finished. ---")

    finally:
        print("--- Stopping CodeCarbon tracker. ---")
        emissions_data = tracker.stop()
        if emissions_data:
            print(f"\nTotal emissions for TRAINING ONLY: {emissions_data} kg CO₂eq")
        else:
            print("No emissions data was captured. The script might have been too short.")
        print(f"Results saved to {output_dir}/emissions.csv")


if __name__ == "__main__":
    main()