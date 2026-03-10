<script setup lang="ts">
import { onMounted } from 'vue';
import { useGameStore } from '../store/game';
import Hand from '../components/Hand.vue';
import Card from '../components/Card.vue';
import type { Card as CardType } from '../types';
import { canBeat, analyzeHand, HandType } from '../lib/rules';

const gameStore = useGameStore();

onMounted(() => {
  gameStore.initGame();
});

const handlePlay = (cards: CardType[]) => {
  // Logic to validate and play cards
  if (!gameStore.humanPlayer) return;

  // Directly call store action (Backend will validate)
  gameStore.playCards(gameStore.humanPlayer.id, cards);
};

const handlePass = () => {
  gameStore.passTurn();
};
</script>

<template>
  <div class="min-h-screen bg-green-800 flex flex-col justify-between p-4 overflow-hidden">
    <!-- Top Area: Opponents -->
    <div class="flex justify-between items-start px-8 pt-16 relative">
      <!-- Auto-play Toggle Button -->
      <button
        @click="gameStore.toggleAutoPlay()"
        class="absolute -top-2 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full font-bold transition-all shadow-lg z-10"
        :class="gameStore.isAutoPlay ? 'bg-red-500 text-white hover:bg-red-600' : 'bg-blue-500 text-white hover:bg-blue-600'"
      >
        {{ gameStore.isAutoPlay ? '取消托管 (Active)' : '开启托管 (Auto-play)' }}
      </button>

      <!-- Opponent 1 (Left) -->
      <div class="flex flex-col items-center relative">
        <div class="w-16 h-16 bg-gray-300 rounded-full mb-2 flex items-center justify-center text-xl font-bold">
          Bot 1
        </div>
        <div class="bg-black/30 text-white px-3 py-1 rounded-full mb-2">
          {{ gameStore.opponent1?.cardsCount || 0 }} cards
        </div>
        <!-- Simplified hand view for opponent -->
        <div class="flex -space-x-12 transform scale-75">
             <div v-for="i in gameStore.opponent1?.cardsCount || 0" :key="i" class="w-16 h-24 bg-blue-700 border border-white rounded shadow-md"></div>
        </div>
        <div v-if="gameStore.opponent1?.isTurn" class="mt-2 text-yellow-300 font-bold animate-pulse">Thinking...</div>

        <!-- Opponent 1's Played Cards Area -->
        <div v-if="gameStore.lastPlayedCardsByPlayer[1]" class="absolute top-60 -right-56 flex justify-center">
          <div
            v-for="(card, index) in gameStore.lastPlayedCardsByPlayer[1]"
            :key="card.id"
            :style="{ marginLeft: index === 0 ? '0' : '-72px', zIndex: index }"
          >
            <Card :card="card" size="normal" />
          </div>
        </div>
      </div>

      <!-- Center Area -->
      <div class="flex flex-col items-center justify-center mt-24">
         <div v-if="gameStore.lastPlayedCards.length === 0" class="text-white/50 text-xl font-bold border-2 border-white/20 p-4 rounded-lg">
            Waiting for play...
         </div>
      </div>

      <!-- Opponent 2 (Right) -->
      <div class="flex flex-col items-center relative">
        <div class="w-20 h-20 bg-gray-300 rounded-full mb-2 flex items-center justify-center text-xl font-bold">
          Bot 2
        </div>
        <div class="bg-black/30 text-white px-3 py-1 rounded-full mb-2">
          {{ gameStore.opponent2?.cardsCount || 0 }} cards
        </div>
        <div class="flex -space-x-12 transform scale-75">
             <div v-for="i in gameStore.opponent2?.cardsCount || 0" :key="i" class="w-16 h-24 bg-blue-700 border border-white rounded shadow-md"></div>
        </div>
        <div v-if="gameStore.opponent2?.isTurn" class="mt-2 text-yellow-300 font-bold animate-pulse">Thinking...</div>

        <!-- Opponent 2's Played Cards Area -->
        <div v-if="gameStore.lastPlayedCardsByPlayer[2]" class="absolute top-60 -left-20 flex justify-center">
          <div
            v-for="(card, index) in gameStore.lastPlayedCardsByPlayer[2]"
            :key="card.id"
            :style="{ marginLeft: index === 0 ? '0' : '-72px', zIndex: index }"
          >
            <Card :card="card" size="normal" />
          </div>
        </div>
      </div>
    </div>

    <!-- Bottom Area: Player -->
    <div class="flex flex-col items-center pb-8 relative min-h-[400px]">
       <!-- Player's Played Cards Area -->
       <div v-if="gameStore.lastPlayedCardsByPlayer[0]" class="-mt-40 flex justify-center">
          <div
            v-for="(card, index) in gameStore.lastPlayedCardsByPlayer[0]"
            :key="card.id"
            :style="{ marginLeft: index === 0 ? '0' : '-30px', zIndex: index }"
          >
            <Card :card="card" size="normal" />
          </div>
       </div>

       <!-- Absolute Positioned Hand and Indicator -->
       <div class="absolute bottom-0 left-1/2 -translate-x-1/2 w-full flex flex-col items-center pb-4">
          <div v-if="gameStore.humanPlayer?.isTurn" class="mb-6 text-yellow-300 text-2xl font-bold animate-bounce whitespace-nowrap z-20">
             Your Turn!
          </div>
          <Hand
            v-if="gameStore.humanPlayer"
            :cards="gameStore.humanPlayer.hand"
            :isHuman="true"
            :isTurn="gameStore.humanPlayer.isTurn"
            @play="handlePlay"
            @pass="handlePass"
          />
       </div>
    </div>

    <!-- Settlement Overlay -->
    <div v-if="gameStore.status === 'settlement'" class="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
        <div class="bg-white p-8 rounded-xl shadow-2xl text-center">
            <h2 class="text-4xl font-bold mb-4" :class="gameStore.winnerId === 'human' ? 'text-green-600' : 'text-red-600'">
                {{ gameStore.winnerId === 'human' ? 'You Win!' : 'Game Over' }}
            </h2>
            <p class="text-xl mb-6">Winner: {{ gameStore.players.find(p => p.id === gameStore.winnerId)?.name }}</p>
            <button
                @click="gameStore.initGame()"
                class="px-8 py-3 bg-blue-600 text-white rounded-full font-bold hover:bg-blue-700 transition"
            >
                Play Again
            </button>
        </div>
    </div>
  </div>
</template>
