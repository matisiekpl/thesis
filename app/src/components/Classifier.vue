<script setup>

import Header from "@/components/Header.vue";
import {ref} from "vue";

const model = ref('efficientnet_b5');

import {useDropZone} from '@vueuse/core'

const dropZoneRef = ref();
const image = ref(null);
const filename =ref(null);

function onDrop(files) {
  const reader = new FileReader();
  reader.onload = function (e) {
    image.value = e.target.result;
  };
  reader.readAsDataURL(files[0]);
  filename.value = files[0].name;
}

const {isOverDropZone} = useDropZone(dropZoneRef, {
  onDrop,
  dataTypes: ['image/jpeg']
})

</script>

<template>

  <div class="h-screen">
    <Header/>

    <div class="grid grid-cols-1 md:grid-cols-3 p-4 gap-4 h-[calc(100vh-64px)]">

      <div class="col-span-2 flex flex-col w-full h-full b">
        <img :src="image" class="object-cover  mb-4 rounded-xl h-[calc(100vh-64px-256px)]" v-if="image" alt="Podgląd" >
        <div
            class="h-full w-full order border-dashed border-2 rounded-xl flex justify-center items-center text-3xl text-gray-500 italic cursor-pointer transition-all"
            :class="{'bg-orange-100':isOverDropZone}"
            ref="dropZoneRef">
          <span v-if="isOverDropZone">Zwolnij przycisk by załadować</span>
          <span v-else>Upuść tutaj rozmaz w PNG lub JPEG</span>
        </div>
      </div>

      <div class="w-full border border-2 rounded-xl p-4">

        <div class="ml-1 mb-2">Wybierz model</div>
        <select v-model="model"
                class="py-3 px-4 pe-9 block w-full border border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none">
          <option selected>Wybierz model</option>
          <option value="efficientnet_b5">EfficientNet B5</option>
        </select>

        <div class="ml-1 mb-2 mt-4">Plik</div>
        <input v-model="filename" readonly class="py-3 px-4 pe-9 block w-full border border-gray-200 rounded-lg text-sm focus:border-blue-500 focus:ring-blue-500 disabled:opacity-50 disabled:pointer-events-none">

        <button :disabled="!image"
                class=" w-full disabled:opacity-50 disabled:outline-none disabled:cursor-not-allowed bg-purple-600 px-5 py-3 text-white rounded-lg font-semibold text-sm hover:outline outline-4 outline-purple-300 transition-all ease-in-out mt-4 cursor-pointer text-center">
          Rozpoznaj rodzaj komórki
        </button>

        <table class="w-full mt-4 ">
          <thead class="font-semibold">
          <tr class=" rounded-lg">
            <td class="px-3 py-2 border">Rodzaj komórki</td>
            <td class="px-3 py-2 border text-right">Prawdopodobieństwo</td>
          </tr>
          </thead>
          <tbody>
          <tr class=" rounded-lg">
            <td class="px-3 py-2 border">Leukocyt</td>
            <td class="px-3 py-2 border text-right">94%</td>
          </tr>
          <tr class=" rounded-lg">
            <td class="px-3 py-2 border">Limfocyt</td>
            <td class="px-3 py-2 border text-right">54%</td>
          </tr>
          </tbody>
        </table>
      </div>

    </div>
  </div>


</template>

<style scoped>

</style>