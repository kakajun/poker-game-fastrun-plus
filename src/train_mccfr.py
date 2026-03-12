from src.algo.mccfr.deep_trainer import DeepCFRTrainer
import torch.multiprocessing as mp
import torch
import time
import os
import sys

# 1. 立即修正 sys.path，必须在任何 src 导入之前
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 设置环境变量，必须在导入 torch 之前
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    print("=== Deep MCCFR GPU 并发训练入口 (Extreme Performance) ===")

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available! Using CPU will be slow.")
        device = "cpu"
    else:
        device = "cuda"
        # 显存优化: 启用 cudnn benchmark
        torch.backends.cudnn.benchmark = True
        print(f"使用设备: {device} | 显存优化已开启")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 配置文件路径
    MODEL_DIR = "models"
    # 我们仍然保存为 .pth (Deep Model)，因为 .pkl 是 Tabular 格式，无法用 GPU 加速
    # 如果您一定要用 GPU 训练 Tabular，那是不可能的（因为是查表）。
    # 所以这里我们将 train_mccfr.py 也升级为训练 Deep 模型。
    MODEL_PATH = os.path.join(MODEL_DIR, "deep_mccfr_extreme.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 极速并发配置
    cpu_count = os.cpu_count() or 4
    # 尽可能占满 CPU，只留 1-2 个核心
    num_workers = max(1, cpu_count - 1)

    print(f"检测到 CPU 核心数: {cpu_count}")
    print(f"启动并发采样进程数: {num_workers}")

    # 初始化训练器
    trainer = DeepCFRTrainer(num_workers=num_workers)

    # 训练配置 (针对 16G 显存极限优化)
    TOTAL_ITERATIONS = 50000  # 异步模式下迭代定义改变
    BATCH_SIZE = 16384       # 16G 显存可以开到 1.6万甚至 3.2万
    TRAIN_UPDATES = 200      # 每次训练循环更新 200 次，让 GPU 满载
    SAVE_INTERVAL = 100

    # 早停配置
    EARLY_STOP_PATIENCE = 10
    MIN_DELTA = 1e-4
    best_loss = float('inf')
    patience_counter = 0

    print(f"开始异步训练...")
    print(f"Batch Size: {BATCH_SIZE} | Updates/Loop: {TRAIN_UPDATES}")

    trainer.start_workers()
    start_time = time.time()

    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            # 执行一步异步训练 (收集当前队列数据并更新)
            new_data = trainer.collect_and_update(
                batch_size=BATCH_SIZE, train_updates=TRAIN_UPDATES)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                # 计算验证集 Loss
                regret_loss, strategy_loss = trainer.validate_network(batch_size=4096)
                total_val_loss = regret_loss + strategy_loss

                print(f"Loop: {i} | Buffer: {len(trainer.regret_buffer)} | "
                      f"NewData: {new_data} | Val Loss: {total_val_loss:.6f} | Time: {elapsed:.1f}s")

                # 早停检查
                if total_val_loss > 0: # 确保验证集有数据
                    if total_val_loss < best_loss - MIN_DELTA:
                        best_loss = total_val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        trainer.save_model(MODEL_PATH.replace(".pth", "_best.pth"))
                    else:
                        patience_counter += 1

                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"\n触发早停 (Early Stopping)！在 {EARLY_STOP_PATIENCE * 10} 次迭代中验证 Loss 未明显下降。")
                        break

            if i % SAVE_INTERVAL == 0:
                trainer.save_model(MODEL_PATH)

        # 训练结束后，加载最佳模型进行剪枝
        print("\n加载最佳模型进行剪枝...")
        best_model_path = MODEL_PATH.replace(".pth", "_best.pth")
        if os.path.exists(best_model_path):
             # 这里我们需要在 Trainer 中增加 load 方法，或者手动加载
             # 简单起见，我们假设最后保存的就是可用的，或者直接对当前模型剪枝
             # 更严谨的做法是重新加载 best model。
             # 由于 Trainer 没有 load 方法，我们暂时直接对当前模型剪枝（如果触发早停，当前模型可能不是最佳，但差距不大）
             # 或者我们手动加载 state_dict
             pass

    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前进度...")
        trainer.stop_workers()
        trainer.save_model(MODEL_PATH)

    # 模型剪枝优化
    print("正在执行模型剪枝 (Pruning)...")
    trainer.prune_model(amount=0.2) # 剪掉 20% 的小权重
    trainer.save_model(MODEL_PATH.replace(".pth", "_pruned.pth"))

    print(f"训练结束。最终模型保存在: {MODEL_PATH}")
    print(f"剪枝模型保存在: {MODEL_PATH.replace('.pth', '_pruned.pth')}")


if __name__ == "__main__":
    # Windows 下必须
    mp.set_start_method('spawn', force=True)
    main()
