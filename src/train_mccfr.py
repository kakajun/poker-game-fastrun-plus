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
                # 统计每秒处理的数据条数
                print(f"Loop: {i} | Buffer: {len(trainer.regret_buffer)} | "
                      f"NewData: {new_data} | Time: {elapsed:.1f}s")

            if i % SAVE_INTERVAL == 0:
                trainer.save_model(MODEL_PATH)

    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前进度...")
        trainer.stop_workers()
        trainer.save_model(MODEL_PATH)

    print(f"训练结束。最终模型保存在: {MODEL_PATH}")


if __name__ == "__main__":
    # Windows 下必须
    mp.set_start_method('spawn', force=True)
    main()
