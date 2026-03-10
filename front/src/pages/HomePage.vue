<script setup lang="ts">
import { useRouter } from 'vue-router';

const router = useRouter();

const startGame = () => {
  router.push('/game');
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
