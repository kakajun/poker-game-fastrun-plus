import unittest
from src.core.game import Game
from src.core.card import Rank, Suit

class TestGameSort(unittest.TestCase):
    def test_hand_sorting(self):
        """测试手牌是否按点数降序、花色降序排列"""
        # 初始化游戏 (固定种子以便复现，虽然排序逻辑应该对任何手牌都有效)
        game = Game(seed=42)
        
        for i, hand in enumerate(game.hands):
            print(f"\nChecking Hand {i}: {hand}")
            
            # 检查每一张牌是否 >= 下一张牌
            for j in range(len(hand) - 1):
                curr_card = hand[j]
                next_card = hand[j+1]
                
                # 比较逻辑：Rank 大优先；Rank 相同则 Suit 小优先 (因为 Suit.SPADE=0, DIAMOND=3)
                # 等等，我们在 Game.py 里用的 sort_key 是 (rank.value, -suit.value)
                # 也就是 Rank 大在前。Rank 相同，-Suit 大在前 => Suit 小在前。
                # Suit: Spade=0, Heart=1, Club=2, Diamond=3
                # -Suit: 0, -1, -2, -3
                # 所以 Spade(0) > Diamond(3) ?
                # 让我们确认一下 sort_key: (rank, -suit) reverse=True
                # 如果 rank 相等：
                # (-0) vs (-3) -> 0 > -3. 
                # 所以 reverse=True 时，(-0) 排在 (-3) 前面。
                # 即 Spade 排在 Diamond 前面。
                # 这符合黑红梅方 (Spade > Heart > Club > Diamond) 的通常顺序吗？
                # 桥牌通常是黑红梅方。
                # 所以 Spade(0) 应该最大。
                
                # 验证 Rank
                if curr_card.rank.value > next_card.rank.value:
                    continue # 正确
                elif curr_card.rank.value == next_card.rank.value:
                    # 验证 Suit
                    # 应该 curr_card.suit.value < next_card.suit.value (数值越小，地位越高)
                    self.assertLess(curr_card.suit.value, next_card.suit.value, 
                                    f"Suit order wrong at index {j}: {curr_card} vs {next_card}")
                else:
                    self.fail(f"Rank order wrong at index {j}: {curr_card} vs {next_card}")

if __name__ == '__main__':
    unittest.main()
