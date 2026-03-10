import axios from 'axios';

const API_BASE_URL = '/api'; // 使用 Vite 代理前缀

// 后端数据模型定义 (参考后端 src/api/models.py)
export interface CardModel {
  rank: number;
  suit: number;
  id: number;
}

export interface PlayModel {
  type: string;
  cards: CardModel[];
  length: number;
  max_rank: number;
}

export interface GameStateModel {
  game_id: string;
  current_player: number;
  hands: CardModel[][];
  last_play: PlayModel | null;
  last_play_player: number;
  winner: number;
  is_over: boolean;
  scores: number[];
  cards_played_count: number[];
  bomb_scores: number[];
  legal_actions: PlayModel[];
}

export const gameApi = {
  async startGame(): Promise<GameStateModel> {
    const response = await axios.post(`${API_BASE_URL}/game/start`);
    return response.data;
  },

  async getState(gameId: string): Promise<GameStateModel> {
    const response = await axios.get(`${API_BASE_URL}/game/${gameId}/state`);
    return response.data;
  },

  async playerAction(gameId: string, cardIds: number[]): Promise<GameStateModel> {
    const response = await axios.post(`${API_BASE_URL}/game/${gameId}/action`, {
      card_ids: cardIds
    });
    return response.data;
  },

  async triggerAi(gameId: string): Promise<GameStateModel> {
    const response = await axios.post(`${API_BASE_URL}/game/${gameId}/ai`);
    return response.data;
  }
};
