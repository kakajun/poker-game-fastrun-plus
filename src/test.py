import numpy as np

class VADCFR:
    def __init__(self, num_actions, volatility_window=50, init_discount=0.95):
        self.num_actions = num_actions  # 博弈中可选动作数
        self.vol_window = volatility_window  # 计算波动的滑动窗口
        self.regret_sum = np.zeros(num_actions)  # 遗憾值累积
        self.strategy_sum = np.zeros(num_actions)  # 策略累积
        self.strategy = np.ones(num_actions) / num_actions  # 当前策略
        self.discount = init_discount  # 初始折扣率
        self.regret_history = []  # 遗憾值历史（用于计算波动）

    def _calculate_volatility(self):
        """核心：计算遗憾值的波动系数（VAD 的关键）"""
        if len(self.regret_history) < self.vol_window:
            return 0.1  # 初始波动默认值
        recent_regrets = np.array(self.regret_history[-self.vol_window:])
        # 波动系数 = 标准差 / 均值（绝对值，避免正负抵消）
        volatility = np.std(recent_regrets) / (np.mean(np.abs(recent_regrets)) + 1e-8)
        return volatility

    def _update_discount(self, volatility):
        """自适应调整折扣率：波动大则提高折扣（快速遗忘），反之降低"""
        # 波动系数范围映射到折扣率 [0.8, 0.99]
        self.discount = 0.8 + 0.19 * (1 - np.clip(volatility, 0, 1))

    def _optimistic_update(self, regret):
        """乐观更新：放大正向遗憾的权重"""
        optimistic_regret = np.maximum(regret, 0) * 1.2  # 正向遗憾加权1.2倍
        return optimistic_regret

    def update_strategy(self, regret):
        """VAD-CFR 核心更新逻辑"""
        # 1. 乐观更新遗憾值
        opt_regret = self._optimistic_update(regret)
        # 2. 记录遗憾值用于计算波动
        self.regret_history.append(np.sum(opt_regret))
        # 3. 计算波动并更新折扣率
        volatility = self._calculate_volatility()
        self._update_discount(volatility)
        # 4. 折扣化累积遗憾（动态折扣）
        self.regret_sum = self.discount * self.regret_sum + opt_regret
        # 5. 更新当前策略
        self.strategy = np.maximum(self.regret_sum, 0)
        self.strategy = self.strategy / np.sum(self.strategy) if np.sum(self.strategy) > 0 else np.ones(self.num_actions)/self.num_actions
        # 6. 硬热启动：前500轮不累积策略，仅累积遗憾
        if len(self.regret_history) > 500:
            self.strategy_sum += self.strategy

    def get_average_strategy(self):
        """获取平均策略（逼近纳什均衡）"""
        if len(self.regret_history) <= 500:
            return self.strategy  # 热启动阶段返回当前策略
        avg_strategy = self.strategy_sum / np.sum(self.strategy_sum)
        return avg_strategy

# ---------------------- 测试示例 ----------------------
if __name__ == "__main__":
    # 模拟 Leduc Poker 中某个决策点（3个可选动作）
    vad_cfr = VADCFR(num_actions=3)
    # 模拟1000轮迭代（输入随机遗憾值，仅作演示）
    for i in range(1000):
        fake_regret = np.random.randn(3)  # 随机生成遗憾值
        vad_cfr.update_strategy(fake_regret)
        if i % 100 == 0:
            print(f"迭代 {i} | 折扣率 {vad_cfr.discount:.3f} | 策略 {vad_cfr.get_average_strategy().round(3)}")