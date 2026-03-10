from src.algo.mccfr.deep_trainer import DeepCFRTrainer
import os
import sys
import time
import torch

# 1. 立即修正 sys.path
# 获取当前文件所在目录的上一级目录（项目根目录）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 现在可以安全导入 src 模块了

# 环境变量设置
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def main():
    print("=== Deep MCCFR GPU 并发训练入口 (V3) ===")

    # 检查 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 配置文件路径
    MODEL_DIR = "models"
    MODEL_PATH = os.path.join(MODEL_DIR, "deep_mccfr_gpu.pth")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 初始化训练器 (默认根据 CPU 核心数调整 num_workers)
    num_workers = min(os.cpu_count() or 4, 8)
    trainer = DeepCFRTrainer(num_workers=num_workers)

    # 训练配置
    TOTAL_ITERATIONS = 1000  # 总训练轮数
    GAMES_PER_WORKER = 5     # 每轮每个 worker 采集的游戏局数
    SAVE_INTERVAL = 50       # 每 50 轮保存一次

    print(f"并发采样进程数: {num_workers}")
    print(f"开始训练...")

    start_time = time.time()

    try:
        for i in range(1, TOTAL_ITERATIONS + 1):
            # 执行一步训练 (采样 + GPU 更新)
            data_count = trainer.train_step(games_per_worker=GAMES_PER_WORKER)

            if i % 10 == 0:
                elapsed = time.time() - start_time
                total_games = i * GAMES_PER_WORKER * num_workers
                fps = total_games / elapsed if elapsed > 0 else 0
                print(f"轮次: {i}/{TOTAL_ITERATIONS} | 缓存大小: {len(trainer.regret_buffer)} | "
                      f"速度: {fps:.1f} games/s | 用时: {elapsed:.1f}s")

            if i % SAVE_INTERVAL == 0:
                trainer.save_model(MODEL_PATH)

    except KeyboardInterrupt:
        print("\n训练被手动中断，正在保存当前进度...")
        trainer.save_model(MODEL_PATH)

    print(f"训练结束。最终模型保存在: {MODEL_PATH}")


if __name__ == "__main__":
    # 多进程在 Windows 下必须在 main 中运行
    main()
