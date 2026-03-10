
import { Rank } from '../types';
import type { Card } from '../types';

export enum HandType {
  Single,
  Pair,
  Triple,
  TripleWithSingle,
  Bomb,
  Straight, // 5+
  StraightPairs, // 3+
  Plane, // 2+ triples
  PlaneWithWings,
  Invalid
}

interface HandPattern {
  type: HandType;
  rank: number; // The rank value for comparison (e.g., rank of the pair, or start of straight)
  length: number; // Number of cards (e.g., 5 for straight)
}

// Helper to get rank counts
function getRankCounts(cards: Card[]) {
  const counts = new Map<Rank, number>();
  for (const card of cards) {
    counts.set(card.rank, (counts.get(card.rank) || 0) + 1);
  }
  return counts;
}

// Helper to sort cards
function sortCards(cards: Card[]) {
  return [...cards].sort((a, b) => a.rank - b.rank);
}

export function analyzeHand(cards: Card[]): HandPattern {
  const sorted = sortCards(cards);
  const len = sorted.length;
  const counts = getRankCounts(sorted);
  const uniqueRanks = Array.from(counts.keys()).sort((a, b) => a - b);

  // Single
  if (len === 1) {
    return { type: HandType.Single, rank: sorted[0].rank, length: 1 };
  }

  // Pair
  if (len === 2 && counts.size === 1) {
    return { type: HandType.Pair, rank: sorted[0].rank, length: 2 };
  }

  // Triple
  if (len === 3 && counts.size === 1) {
    return { type: HandType.Triple, rank: sorted[0].rank, length: 3 };
  }

  // Bomb
  if (len === 4 && counts.size === 1) {
    return { type: HandType.Bomb, rank: sorted[0].rank, length: 4 };
  }

  // Triple with Single
  if (len === 4 && counts.size === 2) {
    // Check if it's 3+1
    for (const [rank, count] of counts.entries()) {
      if (count === 3) {
        return { type: HandType.TripleWithSingle, rank: rank, length: 4 };
      }
    }
  }

  // Straight (5+)
  if (len >= 5 && counts.size === len) {
      // Check for consecutive ranks
      // Note: 2 cannot be in straight usually? Rules don't specify, but usually 2 is high.
      // Standard: 3,4,5,6,7.  A,2 cannot be in straight usually in Run Fast?
      // "3 < 4 < ... < A < 2".
      // Usually straights are 3-A. 2 is not in straight.
      const isConsecutive = uniqueRanks.every((r, i) => {
          if (i === 0) return true;
          return r === uniqueRanks[i-1] + 1;
      });
      // Ensure no 2 in straight (Rank.Two = 15)
      const hasTwo = uniqueRanks.includes(Rank.Two);

      if (isConsecutive && !hasTwo) {
          return { type: HandType.Straight, rank: uniqueRanks[0], length: len };
      }
  }

  // Straight Pairs (3+ pairs)
  // e.g. 334455
  if (len >= 6 && len % 2 === 0 && counts.size === len / 2) {
      const isAllPairs = Array.from(counts.values()).every(c => c === 2);
      if (isAllPairs) {
           const isConsecutive = uniqueRanks.every((r, i) => {
              if (i === 0) return true;
              return r === uniqueRanks[i-1] + 1;
           });
           const hasTwo = uniqueRanks.includes(Rank.Two);
           if (isConsecutive && !hasTwo) {
               return { type: HandType.StraightPairs, rank: uniqueRanks[0], length: len };
           }
      }
  }

  // Plane (2+ triples)
  // e.g. 333444
  // Plane with wings: 333444 + 5 + 6 (8 cards) or 333444 + 55 + 66 (10 cards)?
  // Rules say: "Plane + same number of singles or pairs"
  // Let's implement pure Plane first
  // ... (Simplifying for now, will add more complex patterns if needed)

  return { type: HandType.Invalid, rank: 0, length: 0 };
}

export function canBeat(play: Card[], lastPlay: Card[]): boolean {
  if (lastPlay.length === 0) return true; // Free play (should be handled by game logic, but here for completeness)

  const playPattern = analyzeHand(play);
  const lastPattern = analyzeHand(lastPlay);

  if (playPattern.type === HandType.Invalid) return false;

  // Bomb beats everything except bigger bomb
  if (playPattern.type === HandType.Bomb) {
    if (lastPattern.type !== HandType.Bomb) return true;
    return playPattern.rank > lastPattern.rank;
  }

  // Otherwise types must match and length must match
  if (playPattern.type !== lastPattern.type) return false;
  if (playPattern.length !== lastPattern.length) return false;

  return playPattern.rank > lastPattern.rank;
}
