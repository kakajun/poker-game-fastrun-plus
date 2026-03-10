
<script setup lang="ts">
import { ref, computed } from 'vue';
import type { Card as CardType } from '../types';
import Card from './Card.vue';

const props = defineProps<{
  cards: CardType[];
  isHuman?: boolean;
  isTurn?: boolean;
}>();

const emit = defineEmits<{
  (e: 'play', cards: CardType[]): void;
  (e: 'pass'): void;
}>();

const selectedCards = ref<Set<string>>(new Set());

// Use cards directly as provided by props (assuming backend handles sorting)
const sortedCards = computed(() => {
  if (!props.cards) return [];
  return props.cards;
});

const toggleSelect = (card: CardType) => {
  if (!props.isHuman) return;

  if (selectedCards.value.has(card.id)) {
    selectedCards.value.delete(card.id);
  } else {
    selectedCards.value.add(card.id);
  }
};

const playSelected = () => {
  const cardsToPlay = props.cards.filter(c => selectedCards.value.has(c.id));
  emit('play', cardsToPlay);
  selectedCards.value.clear();
};

const pass = () => {
  emit('pass');
  selectedCards.value.clear();
};

const cancelSelection = () => {
  selectedCards.value.clear();
};

const canPlay = computed(() => {
  return selectedCards.value.size > 0 && props.isTurn;
});
</script>

<template>
  <div class="flex flex-col items-center gap-6 w-full">
    <!-- Controls (Only for Human) -->
    <div v-if="isHuman && isTurn" class="flex gap-4 z-50">
      <button
        @click="pass"
        class="px-6 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-full font-bold shadow-lg transition-colors"
      >
        Pass
      </button>

      <button
        @click="cancelSelection"
        v-if="selectedCards.size > 0"
        class="px-6 py-2 bg-red-500 hover:bg-red-600 text-white rounded-full font-bold shadow-lg transition-colors"
      >
        Cancel
      </button>

      <button
        @click="playSelected"
        :disabled="!canPlay"
        class="px-6 py-2 bg-green-500 hover:bg-green-600 disabled:bg-gray-400 text-white rounded-full font-bold shadow-lg transition-colors disabled:cursor-not-allowed"
      >
        Play
      </button>
    </div>

    <!-- Cards Display (Standard Fan Layout) -->
    <div class="flex justify-center items-end h-62 relative">
      <div
        v-for="(card, index) in sortedCards"
        :key="card.id"
        class="transition-all duration-300"
        :style="{
          marginLeft: index === 0 ? '0' : '-145px', // More compact overlap for larger cards
          zIndex: index
        }"
      >
        <Card
          :card="card"
          :isSelected="selectedCards.has(card.id)"
          :size="isHuman ? 'large' : 'small'"
          @click="toggleSelect(card)"
        />
      </div>
    </div>
  </div>
</template>
