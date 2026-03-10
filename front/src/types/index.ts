export enum Suit {
  Spades = 'S',
  Hearts = 'H',
  Clubs = 'C',
  Diamonds = 'D',
}

export enum Rank {
  Three = 3,
  Four = 4,
  Five = 5,
  Six = 6,
  Seven = 7,
  Eight = 8,
  Nine = 9,
  Ten = 10,
  Jack = 11,
  Queen = 12,
  King = 13,
  Ace = 14,
  Two = 15,
}

export interface Card {
  suit: Suit;
  rank: Rank;
  id: string; // unique identifier for key
}

export interface Player {
  id: string;
  name: string;
  hand: Card[];
  isHuman: boolean;
  score: number;
  cardsCount: number; // to hide AI hand but show count
  isTurn: boolean;
}

export type GameStatus = 'lobby' | 'playing' | 'settlement';

export interface GameState {
  status: GameStatus;
  players: Player[];
  currentPlayerIndex: number;
  lastPlayedCards: Card[];
  lastPlayedCardsByPlayer: (Card[] | null)[]; // 每位玩家最后出的牌
  lastPlayerIndex: number | null; // who played the last cards
  deck: Card[];
  winnerId: string | null;
}
