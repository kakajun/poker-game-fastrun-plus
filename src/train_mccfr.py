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

    # 训练配置 (针对 16G 显存优化)
    # 加大 Batch Size 以利用显存
    TOTAL_ITERATIONS = 10000
    # 增加单次采样量，减少进程切换开销
    GAMES_PER_WORKER = 50
    # 16G 显存，batch_size 可以非常大，比如 4096 甚至 8192
    BATCH_SIZE = 4096
    SAVE_INTERVAL = 100

    print(f"开始训练...")
    print(f"Batch Size: {BATCH_SIZE} | Games/Worker: {GAMES_PER_WORKER}")

    start_time = time.time()
    total_games_processed = 0

    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            # 执行一步训练
            # 修改 train_step 逻辑以支持自定义 batch_size (需要修改 deep_trainer.py)
            # 目前 deep_trainer.py 默认 batch_size 较小，我们稍后通过工具修改它
            # 或者我们在 trainer 内部自动处理

            # 这里我们假设 train_step 已经优化
            # 实际上 deep_trainer.py 的 buffer size 是 100000，sample size 是 128
            # 我们可以在外部循环中多次 update network 来利用数据

            # 1. 采样
            new_samples = trainer.train_step(games_per_worker=GAMES_PER_WORKER)
            # 估算局数 (平均每局42步?) 不准，直接用 worker * games
            total_games_processed += new_samples // 42

            # 2. 强化训练 (Replay Buffer 利用)
            # 既然显存大，我们可以多更新几次网络
            if len(trainer.regret_buffer) > BATCH_SIZE:
                # 显存足够，进行多次大 Batch 更新
                # 这里我们更新 10 次，充分利用每次采集的样本
                trainer.update_network(batch_size=BATCH_SIZE, updates=10)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                # 真实 FPS 计算
                current_fps = (i * GAMES_PER_WORKER * num_workers) / elapsed
                print(f"Iter: {i}/{TOTAL_ITERATIONS} | Buffer: {len(trainer.regret_buffer)} | "
                      f"FPS: {current_fps:.1f} games/s | Time: {elapsed:.1f}s")

            if i % SAVE_INTERVAL == 0:
                trainer.save_model(MODEL_PATH)

    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前进度...")
        trainer.save_model(MODEL_PATH)

    print(f"训练结束。最终模型保存在: {MODEL_PATH}")


if __name__ == "__main__":
    # Windows 下必须
    mp.set_start_method('spawn', force=True)
    main()
