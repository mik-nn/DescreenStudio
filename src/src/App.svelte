<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";

  interface PreviewOut {
    data_url: string;
    width: number;
    height: number;
  }

  interface DescreenOut {
    output_path: string;
    width: number;
    height: number;
    elapsed_ms: number;
  }

  interface ProgressPayload {
    done: number;
    total: number;
  }

  let aiBlend = $state(0.85);
  let isProcessing = $state(false);
  let isLoading = $state(false);
  let imagePath = $state<string | null>(null);
  let beforeUrl = $state<string | null>(null);
  let afterUrl = $state<string | null>(null);
  let spectrumUrl = $state<string | null>(null);
  let dims = $state("");
  let status = $state("Откройте скан (PNG / TIFF / JPEG)…");
  let outputPath = $state<string | null>(null);
  let progress = $state<ProgressPayload | null>(null);
  let splitPos = $state(0.5);

  let canvas = $state<HTMLCanvasElement | null>(null);
  let beforeImg: HTMLImageElement | null = null;
  let afterImg: HTMLImageElement | null = null;

  function loadImageEl(url: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = () => reject(new Error("preview decode failed"));
      el.src = url;
    });
  }

  function drawViewport() {
    if (!canvas || !beforeImg) return;
    const box = canvas.parentElement?.getBoundingClientRect();
    if (!box || box.width < 2 || box.height < 2) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(box.width * dpr);
    canvas.height = Math.floor(box.height * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const s = Math.min(
      canvas.width / beforeImg.naturalWidth,
      canvas.height / beforeImg.naturalHeight,
    );
    const dw = beforeImg.naturalWidth * s;
    const dh = beforeImg.naturalHeight * s;
    const ox = (canvas.width - dw) / 2;
    const oy = (canvas.height - dh) / 2;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(beforeImg, ox, oy, dw, dh);
    if (afterImg) {
      const sx = ox + dw * splitPos;
      ctx.save();
      ctx.beginPath();
      ctx.rect(sx, oy, Math.max(0, ox + dw - sx), dh);
      ctx.clip();
      ctx.drawImage(afterImg, ox, oy, dw, dh);
      ctx.restore();
      ctx.strokeStyle = "#818cf8";
      ctx.lineWidth = 2 * dpr;
      ctx.beginPath();
      ctx.moveTo(sx, oy);
      ctx.lineTo(sx, oy + dh);
      ctx.stroke();
    }
  }

  function onPointer(e: PointerEvent) {
    if (!canvas || !beforeImg || !afterImg) return;
    const r = canvas.getBoundingClientRect();
    const s = Math.min(
      r.width / beforeImg.naturalWidth,
      r.height / beforeImg.naturalHeight,
    );
    const dw = beforeImg.naturalWidth * s;
    const ox = (r.width - dw) / 2;
    splitPos = Math.min(0.995, Math.max(0.005, (e.clientX - r.left - ox) / dw));
    drawViewport();
  }

  async function openScan() {
    if (isLoading || isProcessing) return;
    const sel = await open({
      multiple: false,
      filters: [
        { name: "Сканы", extensions: ["png", "tif", "tiff", "jpg", "jpeg", "bmp"] },
      ],
    });
    if (typeof sel !== "string" || !sel) return;
    isLoading = true;
    imagePath = sel;
    afterImg = null;
    afterUrl = null;
    outputPath = null;
    progress = null;
    try {
      status = "Загрузка превью…";
      const pv = await invoke<PreviewOut>("preview_image", {
        path: sel,
        maxDim: 1600,
      });
      beforeUrl = pv.data_url;
      dims = `${pv.width}×${pv.height}`;
      beforeImg = await loadImageEl(beforeUrl);
      drawViewport();
      status = "Расчёт спектра…";
      const sp = await invoke<PreviewOut>("spectrum_image", { path: sel });
      spectrumUrl = sp.data_url;
      status = `Готов: ${sel}`;
    } catch (err) {
      status = `Ошибка загрузки: ${err}`;
    } finally {
      isLoading = false;
    }
  }

  async function runDescreenProcess() {
    if (!imagePath || isProcessing) return;
    isProcessing = true;
    progress = { done: 0, total: 1 };
    const unlisten = await listen<ProgressPayload>("descreen-progress", (ev) => {
      progress = ev.payload;
    });
    try {
      status = "Дескрининг на GPU…";
      const res = await invoke<DescreenOut>("run_descreen", {
        path: imagePath,
        blend: aiBlend,
      });
      outputPath = res.output_path;
      const pv = await invoke<PreviewOut>("preview_image", {
        path: res.output_path,
        maxDim: 1600,
      });
      afterUrl = pv.data_url;
      afterImg = await loadImageEl(afterUrl);
      drawViewport();
      status = `Готово за ${(res.elapsed_ms / 1000).toFixed(1)} с → ${res.output_path}`;
    } catch (err) {
      status = `Ошибка инференса: ${err}`;
    } finally {
      unlisten();
      isProcessing = false;
    }
  }

  const progressPct = $derived(
    progress && progress.total > 0
      ? Math.min(100, (100 * progress.done) / progress.total)
      : 0,
  );

  $effect(() => {
    const onResize = () => drawViewport();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  });
</script>

<main class="flex h-screen bg-neutral-900 text-neutral-100 font-sans">
  <section class="flex-1 flex flex-col border-r border-neutral-800 min-w-0">
    <header
      class="h-12 border-b border-neutral-800 flex items-center px-4 justify-between shrink-0"
    >
      <span
        class="text-xs font-semibold uppercase tracking-wider text-neutral-400"
        >Preview Viewport</span
      >
      {#if dims}
        <span class="text-xs text-neutral-500">{dims}</span>
      {/if}
    </header>
    <div
      class="flex-1 bg-black flex items-center justify-center relative overflow-hidden"
    >
      {#if beforeUrl}
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div
          class="absolute inset-0 cursor-ew-resize"
          onpointerdown={onPointer}
          onpointermove={(e) => {
            if (e.buttons & 1) onPointer(e);
          }}
        >
          <canvas bind:this={canvas} class="w-full h-full"></canvas>
        </div>
        <div
          class="absolute top-2 left-2 text-[11px] px-2 py-0.5 rounded bg-black/60 text-neutral-300 pointer-events-none"
        >
          До · слева
        </div>
        {#if afterUrl}
          <div
            class="absolute top-2 right-2 text-[11px] px-2 py-0.5 rounded bg-indigo-600/80 text-white pointer-events-none"
          >
            После · справа (тяните разделитель)
          </div>
        {/if}
      {:else}
        <button
          onclick={openScan}
          disabled={isLoading}
          class="px-5 py-3 rounded-md bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 text-sm transition-colors cursor-pointer"
        >
          {isLoading ? "Загрузка…" : "Открыть скан…"}
        </button>
      {/if}
    </div>
    <footer
      class="border-t border-neutral-800 px-4 py-2 text-xs text-neutral-400 shrink-0"
    >
      {#if isProcessing && progress}
        <div class="flex items-center gap-3">
          <div class="flex-1 h-1.5 rounded bg-neutral-800 overflow-hidden">
            <div
              class="h-full bg-indigo-500 transition-all"
              style="width: {progressPct}%"
            ></div>
          </div>
          <span class="tabular-nums whitespace-nowrap"
            >тайл {progress.done}/{progress.total}</span
          >
        </div>
      {:else}
        <span class="block truncate" title={status}>{status}</span>
      {/if}
    </footer>
  </section>

  <aside class="w-80 flex flex-col bg-neutral-950 p-4 gap-6 shrink-0 overflow-y-auto">
    <div>
      <h3 class="text-sm font-medium mb-3 text-neutral-300">
        Частотный спектр (2D FFT)
      </h3>
      <div
        class="aspect-square bg-neutral-900 rounded-lg border border-neutral-800 flex items-center justify-center overflow-hidden"
      >
        {#if spectrumUrl}
          <img
            src={spectrumUrl}
            alt="2D FFT спектр"
            class="w-full h-full object-contain"
          />
        {:else}
          <span class="text-xs text-neutral-600">нет изображения</span>
        {/if}
      </div>
    </div>

    <div class="flex flex-col gap-4">
      <h3 class="text-sm font-medium text-neutral-300">Параметры нейросети</h3>

      <label class="flex flex-col gap-1.5 text-xs text-neutral-400">
        <span>Сила подавления растра (AI Blend): {Math.round(aiBlend * 100)}%</span
        >
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          bind:value={aiBlend}
          class="accent-indigo-500"
        />
      </label>

      <button
        onclick={openScan}
        disabled={isLoading || isProcessing}
        class="py-2 px-4 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-50 rounded-md text-sm transition-colors cursor-pointer"
      >
        Открыть скан…
      </button>

      <button
        onclick={runDescreenProcess}
        disabled={isProcessing || isLoading || !imagePath}
        class="py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-neutral-800 rounded-md font-medium text-sm transition-colors cursor-pointer flex items-center justify-center gap-2"
      >
        {#if isProcessing}
          <span>Обработка на GPU…</span>
        {:else}
          <span>Запустить дескрининг</span>
        {/if}
      </button>

      {#if outputPath}
        <p class="text-[11px] text-neutral-500 break-all">
          Результат: {outputPath}
        </p>
      {/if}
    </div>
  </aside>
</main>
