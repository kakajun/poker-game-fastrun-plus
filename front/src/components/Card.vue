<script setup lang="ts">
import { computed } from 'vue';
import { Suit, Rank } from '../types';
import type { Card } from '../types';
import { cn } from '../lib/utils';

const props = defineProps<{
  card: Card;
  isSelected?: boolean;
  size?: 'small' | 'normal' | 'large';
  isHidden?: boolean; // For opponent cards (back of card)
}>();

const sizeClasses = computed(() => {
  switch (props.size) {
    case 'small': return 'w-16 h-24';
    case 'large': return 'w-44 h-64';
    default: return 'w-24 h-36'; // normal
  }
});

const emit = defineEmits<{
  (e: 'click'): void;
}>();

// Map card to image file index based on the specific image resources:
// 1-4.png: Ace (Rank 14)
// 5-8.png: 2 (Rank 15)
// 9-12.png: 3 (Rank 3)
// ...
// 49-52.png: King (Rank 13)
const imageIndex = computed(() => {
  // 直接从后端传来的原始 ID 进行转换
  const id = parseInt(props.card.id);

  if (id < 44) {
    // Ranks 3-13: 后端 ID 0-43 -> 图片 9-52
    return id + 9;
  } else if (id < 48) {
    // Rank A: 后端 ID 44-47 -> 图片 1-4
    return (id - 44) + 1;
  } else {
    // Rank 2: 后端 ID 48-51 -> 图片 5-8
    return (id - 48) + 5;
  }
});

const imageSrc = computed(() => {
  // Use new URL to resolve asset path
  return new URL(`../assets/porerImg/${imageIndex.value}.png`, import.meta.url).href;
});


const handleClick = () => {
  if (!props.isHidden) {
    emit('click');
  }
};
</script>

<template>
  <div
    class="relative transition-transform duration-200 select-none"
    :class="cn(
      'cursor-pointer hover:-translate-y-2',
      isSelected ? '-translate-y-6' : '',
      sizeClasses,
      isHidden ? 'bg-blue-800 border-2 border-white rounded-lg' : ''
    )"
    @click="handleClick"
  >
    <img
      v-if="!isHidden"
      :src="imageSrc"
      :alt="`${card.suit}${card.rank}`"
      class="w-full h-full object-contain drop-shadow-md"
      draggable="false"
    />
    <div v-else class="w-full h-full flex items-center justify-center text-white">
       <!-- Card Back Pattern -->
       <div class="w-full h-full bg-gradient-to-br from-blue-600 to-blue-900 rounded-lg border border-blue-400"></div>
    </div>
  </div>
</template>
