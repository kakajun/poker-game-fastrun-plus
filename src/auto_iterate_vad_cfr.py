import os
import sys
import time
import torch
import torch.multiprocessing as mp
import numpy as np

# 1. Fix sys.path immediately, before any src imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from src
from src.algo.vad_cfr.vad_trainer import VADCFRTrainer
from src.evaluate.evaluator import ModelEvaluator

# 2. Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_latest_version(model_dir):
    versions = []
    if not os.path.exists(model_dir):
        return 0
    for f in os.listdir(model_dir):
        if f.startswith("v") and f.endswith(".pth"):
            try:
                # Expecting v1.pth, v2.pth, etc.
                v_str = f[1:-4]
                if v_str.isdigit():
                    versions.append(int(v_str))
            except ValueError:
                continue
    return max(versions) if versions else 0

def main():
    print("=== VAD-CFR Auto-Iterative Training & Evaluation ===")
    
    # Configuration
    MODEL_DIR = "models"
    BASE_MODEL = os.path.join(MODEL_DIR, "vad_cfr_best.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Concurrency Configuration
    cpu_count = os.cpu_count() or 4
    num_workers = max(1, cpu_count - 2) # Leave 2 cores for other tasks
    
    # Training Hyperparameters
    BATCH_SIZE = 32768 # Increased batch size for better stability
    TRAIN_UPDATES = 300 # More updates per loop
    ITER_PER_EVAL = 30 # Evaluation frequency
    EVAL_EPISODES = 150 # More games for accurate win rate
    WIN_RATE_TARGET = 0.80

    # Initialize VAD-CFR Trainer
    trainer = VADCFRTrainer(num_workers=num_workers)
    
    # Find current version
    current_v = get_latest_version(MODEL_DIR)
    if current_v > 0:
        champion_path = os.path.join(MODEL_DIR, f"v{current_v}.pth")
        print(f"Latest version found: v{current_v}. Using as champion.")
    else:
        champion_path = BASE_MODEL
        print(f"No versioned models found. Using base model: {BASE_MODEL}")

    if champion_path and not os.path.exists(champion_path):
        print(f"Warning: Champion model {champion_path} not found! Will start from scratch.")
        champion_path = None
    elif champion_path:
        # Load the champion to continue training it as the new candidate
        trainer.load_model(champion_path)

    trainer.start_workers()
    evaluator = ModelEvaluator(n_episodes=EVAL_EPISODES)
    
    iteration = 0
    try:
        while True:
            iteration += 1
            print(f"\n--- Iteration Loop {iteration} ---")
            
            # 1. Training Phase
            print(f"Training for {ITER_PER_EVAL} loops...")
            for i in range(1, ITER_PER_EVAL + 1):
                new_data = trainer.collect_and_update(
                    batch_size=BATCH_SIZE, train_updates=TRAIN_UPDATES)
                
                if i % 10 == 0:
                    regret_loss, strategy_loss = trainer.validate_network(batch_size=4096)
                    status = "WarmStart" if trainer.iteration < trainer.warm_start_iter else "Training"
                    print(f"  [{status}] Loop {i}/{ITER_PER_EVAL} | NewData: {new_data} | Loss: {regret_loss+strategy_loss:.6f}")

            # Save candidate model
            candidate_path = os.path.join(MODEL_DIR, "candidate_vad_cfr.pth")
            trainer.save_model(candidate_path)
            
            # 2. Evaluation Phase
            if champion_path and os.path.exists(champion_path):
                print(f"Evaluating Candidate vs Champion: {os.path.basename(champion_path)}")
                summary = evaluator.evaluate_model(candidate_path, "Candidate", champion_path)
                
                if summary:
                    win_rate = summary["Win Rate"]
                    print(f"Evaluation Results: Win Rate = {win_rate:.2%}")
                    
                    if win_rate >= WIN_RATE_TARGET:
                        current_v += 1
                        new_version_path = os.path.join(MODEL_DIR, f"v{current_v}.pth")
                        if os.path.exists(new_version_path): os.remove(new_version_path)
                        os.rename(candidate_path, new_version_path)
                        champion_path = new_version_path
                        print(f"!!! SUCCESS !!! New Champion: v{current_v}.pth (Win Rate: {win_rate:.2%})")
                        print(f"Iterative process finished as target {WIN_RATE_TARGET:.0%} reached.")
                        break
                    else:
                        print(f"Win rate {win_rate:.2%} < {WIN_RATE_TARGET:.0%}. Continuing training...")
            else:
                # First time, no champion to compare with
                current_v += 1
                champion_path = os.path.join(MODEL_DIR, f"v{current_v}.pth")
                if os.path.exists(champion_path): os.remove(champion_path)
                os.rename(candidate_path, champion_path)
                print(f"First model version created: v{current_v}.pth. Now starting iteration...")
                
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        trainer.stop_workers()
        print("\nIterative training process stopped.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
