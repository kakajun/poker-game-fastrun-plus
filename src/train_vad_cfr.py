import os
import sys
import time
import torch
import torch.multiprocessing as mp

# 1. Fix sys.path immediately, before any src imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from src
from src.algo.vad_cfr.vad_trainer import VADCFRTrainer

# 2. Set environment variables
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    print("=== VAD-CFR (Volatility-Adaptive Discounted CFR) Training Entry ===")

    # Check CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available! Using CPU will be slow.")
        device = "cpu"
    else:
        device = "cuda"
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device} | Memory Optimization: ON")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model Configuration
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "vad_cfr.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Concurrency Configuration
    cpu_count = os.cpu_count() or 4
    num_workers = max(1, cpu_count - 1)

    print(f"CPU Cores: {cpu_count}")
    print(f"Worker Processes: {num_workers}")

    # Initialize VAD-CFR Trainer
    trainer = VADCFRTrainer(num_workers=num_workers)

    # Training Hyperparameters
    TOTAL_ITERATIONS = 50000
    BATCH_SIZE = 16384
    TRAIN_UPDATES = 200
    SAVE_INTERVAL = 100

    # Early Stopping
    EARLY_STOP_PATIENCE = 20 # VAD-CFR might be more volatile initially
    MIN_DELTA = 1e-4
    best_loss = float('inf')
    patience_counter = 0

    print(f"Starting VAD-CFR Asynchronous Training...")
    print(f"Batch Size: {BATCH_SIZE} | Updates/Loop: {TRAIN_UPDATES}")
    print(f"Warm Start Iteration: {trainer.warm_start_iter}")

    trainer.start_workers()
    start_time = time.time()

    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            # Collect data and update networks
            new_data = trainer.collect_and_update(
                batch_size=BATCH_SIZE, train_updates=TRAIN_UPDATES)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                regret_loss, strategy_loss = trainer.validate_network(batch_size=4096)
                total_val_loss = regret_loss + strategy_loss
                
                # Check warm start status
                status = "WarmStart" if trainer.iteration < trainer.warm_start_iter else "Training"

                print(f"Loop: {i} [{status}] | Buffer(R): {len(trainer.regret_buffer)} | Buffer(S): {len(trainer.strategy_buffer)} | "
                      f"NewData: {new_data} | Val Loss: {total_val_loss:.6f} | Volatility: {trainer.volatility_ewma:.4f} | Time: {elapsed:.1f}s")

                # Early Stopping Logic (Only apply after warm start)
                if i > trainer.warm_start_iter and total_val_loss > 0:
                    if total_val_loss < best_loss - MIN_DELTA:
                        best_loss = total_val_loss
                        patience_counter = 0
                        trainer.save_model(MODEL_PATH.replace(".pth", "_best.pth"))
                    else:
                        patience_counter += 1

                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"\nEarly Stopping triggered! No improvement for {EARLY_STOP_PATIENCE * 10} iterations.")
                        break

            if i % SAVE_INTERVAL == 0:
                trainer.save_model(MODEL_PATH)

        # Post-training Pruning
        print("\nTraining finished. Loading best model for pruning...")
        # (Assuming current state is good enough or manually load best)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving progress...")
        trainer.stop_workers()
        trainer.save_model(MODEL_PATH)

    # Pruning
    print("Executing Model Pruning (VAD-CFR)...")
    trainer.prune_model(amount=0.2)
    trainer.save_model(MODEL_PATH.replace(".pth", "_pruned.pth"))

    print(f"Final Model saved to: {MODEL_PATH}")
    print(f"Pruned Model saved to: {MODEL_PATH.replace('.pth', '_pruned.pth')}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
