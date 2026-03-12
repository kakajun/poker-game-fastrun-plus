<script setup lang="ts">
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { gameApi } from '../api/game';

const router = useRouter();
const models = ref<string[]>([]);
const selectedModel = ref<string>('');

onMounted(async () => {
  try {
    models.value = await gameApi.getModels();
    if (models.value.length > 0) {
      selectedModel.value = models.value[0];
    }
  } catch (e) {
    console.error("Failed to load models", e);
  }
});

const startGame = () => {
  router.push({ path: '/game', query: { model: selectedModel.value } });
};
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-green-800 to-green-900 flex flex-col items-center justify-center text-white p-4">
    <div class="text-center mb-12">
      <h1 class="text-6xl font-bold mb-4 drop-shadow-lg">Run Fast Poker</h1>
      <p class="text-xl opacity-80">3-Player Classic Card Game</p>
    </div>

    <div class="bg-white/10 backdrop-blur-md p-8 rounded-2xl shadow-xl border border-white/20 max-w-md w-full">
      <div class="space-y-4">

        <div v-if="models.length > 0">
          <label class="block text-left text-sm font-bold mb-2 opacity-80">Select AI Model:</label>
          <select v-model="selectedModel" class="w-full p-3 bg-black/20 border border-white/20 rounded-lg text-white focus:outline-none focus:border-yellow-500 transition-colors">
            <option v-for="m in models" :key="m" :value="m" class="bg-gray-800">{{ m }}</option>
          </select>
        </div>

        <button
          @click="startGame"
          class="w-full py-4 bg-yellow-500 hover:bg-yellow-400 text-black font-bold text-xl rounded-lg shadow-lg transition-all transform hover:scale-105 active:scale-95"
        >
          Start Game
        </button>

        <div class="text-center text-sm opacity-60 mt-4">
          <p>Stage 1: Core Engine Demo</p>
          <p>Play against simple bots locally.</p>
        </div>
      </div>
    </div>
  </div>
</template>
