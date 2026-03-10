import { defineStore } from 'pinia';
import { Suit, Rank } from '../types';
import type { GameState, Player, Card, GameStatus } from '../types';
import { gameApi, type GameStateModel, type CardModel } from '../api/game';

// Adapter helpers
function mapSuit(backendSuit: number): Suit {
  // Backend: 0: Diamond, 1: Club, 2: Heart, 3: Spade
  switch (backendSuit) {
    case 0: return Suit.Diamonds;
    case 1: return Suit.Clubs;
    case 2: return Suit.Hearts;
    case 3: return Suit.Spades;
    default: return Suit.Spades;
  }
}

function mapCard(c: CardModel): Card {
  return {
    rank: c.rank as Rank,
    suit: mapSuit(c.suit),
    id: c.id.toString() // Convert int ID to string for frontend key
  };
}

export const useGameStore = defineStore('game', {
  state: (): GameState & { gameId: string | null, isAutoPlay: boolean } => ({
    gameId: null,
    isAutoPlay: false,
    status: 'lobby',
    players: [],
    currentPlayerIndex: -1,
    lastPlayedCards: [],
    lastPlayedCardsByPlayer: [null, null, null],
    lastPlayerIndex: null,
    deck: [],
    winnerId: null,
  }),

  getters: {
    currentPlayer: (state) => state.players[state.currentPlayerIndex],
    humanPlayer: (state) => state.players.find(p => p.isHuman),
    opponent1: (state) => state.players.find(p => !p.isHuman && p.id === 'ai-1'),
    opponent2: (state) => state.players.find(p => !p.isHuman && p.id === 'ai-2'),
  },

  actions: {
    async initGame() {
      try {
        const state = await gameApi.startGame();
        this.gameId = state.game_id;
        this.syncState(state);

        // If first player is AI, trigger AI
        if (!this.players[this.currentPlayerIndex].isHuman) {
          this.processAiTurn();
        }
      } catch (e) {
        console.error("Failed to start game", e);
      }
    },

    syncState(backendState: GameStateModel) {
      // 1. Status
      this.status = backendState.is_over ? 'settlement' : 'playing';
      this.winnerId = backendState.winner !== -1 ? (backendState.winner === 0 ? 'human' : `ai-${backendState.winner}`) : null;
      this.currentPlayerIndex = backendState.current_player;
      this.lastPlayerIndex = backendState.last_play_player; // If -1 means no one played yet?

      // 2. Last Played Cards
      if (backendState.last_play) {
        const cards = backendState.last_play.cards.map(mapCard);
        this.lastPlayedCards = cards;

        // 更新对应玩家的出牌记录
        if (backendState.last_play_player !== -1) {
          // 如果是新的一轮（上家出的牌被清空，或上家就是自己），则可能需要清除其他人的显示
          // 但后端逻辑中，如果当前玩家是自由出牌，last_play 会是 null。
          // 这里我们简单处理：只更新最后一次出牌人的显示。
          this.lastPlayedCardsByPlayer[backendState.last_play_player] = cards;
        }
      } else {
        this.lastPlayedCards = [];
        // 如果后端 last_play 为空，说明是新的一轮，清除所有人的显示
        this.lastPlayedCardsByPlayer = [null, null, null];
      }

      // 3. Players
      // Map 0 -> human, 1 -> ai-1, 2 -> ai-2
      const playerIds = ['human', 'ai-1', 'ai-2'];
      const playerNames = ['You', 'Bot 1', 'Bot 2'];

      this.players = backendState.hands.map((hand, index) => {
        const isHuman = index === 0;
        const id = playerIds[index];
        const mappedHand = hand.map(mapCard);

        return {
          id,
          name: playerNames[index],
          hand: mappedHand,
          isHuman,
          score: backendState.scores[index],
          cardsCount: hand.length,
          isTurn: index === backendState.current_player
        };
      });

    // 4. Auto-pass logic
    // If it's human turn and the only legal action is PASS, then pass automatically
    if (!backendState.is_over && backendState.current_player === 0) {
      const isOnlyPass = backendState.legal_actions.length === 1 &&
                         backendState.legal_actions[0].type === 'PASS';

      if (isOnlyPass) {
        console.log("No legal moves available. Auto-passing...");
        // Delay a bit for visual clarity
        setTimeout(() => {
          this.passTurn();
        }, 1000);
      } else if (this.isAutoPlay) {
        // Human is in AutoPlay mode
        console.log("Auto-play mode active. AI deciding for human in 3s...");
        setTimeout(async () => {
          // Re-check if it's still my turn and auto-play is still on
          if (this.currentPlayerIndex === 0 && this.isAutoPlay && !this.winnerId) {
            try {
              const newState = await gameApi.triggerAi(this.gameId!);
              this.syncState(newState);

              // If next is AI (bot), it will be handled by processAiTurn triggered inside syncState?
              // No, syncState doesn't trigger bot AI automatically except in initGame.
              // We need to ensure bots move after human auto-play.
              if (this.currentPlayerIndex !== 0 && !this.winnerId) {
                this.processAiTurn();
              }
            } catch (e) {
              console.error("Auto-play AI failed", e);
            }
          }
        }, 3000); // 3s delay as requested
      }
    }
  },

  toggleAutoPlay() {
    this.isAutoPlay = !this.isAutoPlay;
    // If we just turned it on and it's our turn, trigger it
    if (this.isAutoPlay && this.currentPlayerIndex === 0 && !this.winnerId) {
      // Triggering syncState again to hit the logic above
      // Or just call triggerAi directly?
      // Re-running logic is safer.
      const human = this.players[0];
      if (human && human.isTurn) {
          // We can't easily re-run syncState without data,
          // so let's just wait for the next turn or trigger once here if needed.
          // Actually, if we just turned it on, we should probably trigger the 3s timer.
          console.log("Auto-play enabled during turn. Starting 3s timer...");
          setTimeout(async () => {
            if (this.currentPlayerIndex === 0 && this.isAutoPlay && !this.winnerId) {
                const newState = await gameApi.triggerAi(this.gameId!);
                this.syncState(newState);
                if (this.currentPlayerIndex !== 0 && !this.winnerId) {
                    this.processAiTurn();
                }
            }
          }, 3000);
      }
    }
  },

    async playCards(playerId: string, cards: Card[]) {
      if (!this.gameId) return;

      // Human action
      // Convert cards to IDs
      const cardIds = cards.map(c => parseInt(c.id));

      try {
        const newState = await gameApi.playerAction(this.gameId, cardIds);
        this.syncState(newState);

        // Check if next is AI
        if (!this.players[this.currentPlayerIndex].isHuman && !this.winnerId) {
          this.processAiTurn();
        }
      } catch (e) {
        console.error("Play failed", e);
        // TODO: Show error toast
        alert("Illegal move or server error");
      }
    },

    async passTurn() {
      if (!this.gameId) return;
      // Pass = empty cards
      try {
        const newState = await gameApi.playerAction(this.gameId, []);
        this.syncState(newState);

        if (!this.players[this.currentPlayerIndex].isHuman && !this.winnerId) {
          this.processAiTurn();
        }
      } catch (e) {
        console.error("Pass failed", e);
        alert("Cannot pass (Must beat if possible)");
      }
    },

    async processAiTurn() {
      if (!this.gameId || this.winnerId) return;

      // Delay for visual effect
      setTimeout(async () => {
        try {
          const newState = await gameApi.triggerAi(this.gameId);
          this.syncState(newState);

          // If still AI turn (next player is AI), trigger again
          if (!this.players[this.currentPlayerIndex].isHuman && !this.winnerId) {
            this.processAiTurn();
          }
        } catch (e) {
          console.error("AI failed", e);
        }
      }, 800); // 800ms delay
    }
  }
});
