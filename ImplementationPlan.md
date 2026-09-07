# Техническое задание (Implementation Plan)
## Разработка кроссплатформенного приложения AI Fourier-UNet Descreening Studio

---

## 1. Общее описание системы

**Цель проекта:** Создание высокопроизводительного, портативного кроссплатформенного приложения для устранения полиграфического растра (дескрининга), муара и сканерных шумов с сохранением мелких деталей и четкости текста/контуров.

**Ключевой стек:**
* **Dataset Synthesis & AI Training:** PyTorch 2.x, PyTorch Lightning, ONNX / ONNX Runtime.
* **Core Engine (Backend):** Rust (Tauri v2 Core), `ort` (ONNX Runtime Rust binding), `ndarray`, `image`, `rayon`.
* **Hardware Acceleration:** DirectML (Windows), CoreML / Metal (macOS), CUDA / Vulkan (Linux).
* **GUI (Frontend):** Svelte 5, Tailwind CSS v4, HTML5 Canvas / WebGL.

---

## 2. Архитектура системы

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Front-End (Svelte 5 + Tailwind)                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Canvas Viewport  │  │ FFT Spectrogram  │  │ Controls & Analytics │  │
│  └─────────┬────────┘  └─────────┬────────┘  └──────────┬───────────┘  │
└────────────┼─────────────────────┼──────────────────────┼──────────────┘
             │ Zero-Copy IPC       │ Binary Buffer        │ Events
┌────────────┼─────────────────────┼──────────────────────┼──────────────┐
│            ▼                     ▼                      ▼              │
│                        Tauri v2 / Rust Core Engine                     │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Image Pipeline: Tiling Engine (Hann Overlap-Add) & Color Spaces  │  │
│  └───────────────┬──────────────────────────────────────────────────┘  │
│                  │ Sub-patches [B, C, H, W]                            │
│  ┌───────────────▼──────────────────────────────────────────────────┐  │
│  │ ONNX Runtime Rust (`ort`) + Execution Providers                  │  │
│  │ DirectML (Win)  │  CoreML (macOS)  │  CUDA/Vulkan (Linux)            │  │
│  └───────────────┬──────────────────────────────────────────────────┘  │
└──────────────────┼─────────────────────────────────────────────────────┘
                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   AI Model: Fourier-UNet (FFC-UNet)                    │
│             Spatial-Spectral Hybrid Feature Processing                 │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Фаза 1: Генерация и синтез датасета

Для обучения сети требуется парный датасет: **[Идеальное изображение (Ground Truth) ↔ Изображение с искусственным полиграфическим растром и дефектами сканирования]**.

### 3.1. Источники чистых изображений (Ground Truth)
* **Датасеты:** DIV2K, Flickr2K, HQ-Text/Document datasets (для сохранения читаемости мелких шрифтов).
* **Разрешение:** Минумум 2048×2048 пикселей, без сжатия (PNG / Uncompressed TIFF).
* **Объем:** ~10,000 высококачественных кропов (512×512) для обучения, 1,000 для валидации.

### 3.2. Математический симулятор полиграфического растра (Halftone Synthesizer)
Синтез растра должен моделировать реальные физические процессы офсетной, глубокой и флексографской печати:

1. **AM-растрирование (Regular Halftone Screen):**
   * Разделение на каналы **CMYK** (с виртуальным профайлингом RGB → CMYK).
   * **Поворот углов растра:** Моделирование стандартных углов (C: 15°, M: 75°, Y: 0°, K: 45°).
   * **Линиатура (LPI):** Диапазон от 85 LPI (газетная печать) до 175 LPI (высококачественные журналы).
   * **Форма растровой точки:** Круг, эллипс, квадрат.
   * **Растискивание (Tone Value Increase / Dot Gain):** Моделирование нелинейного расширения жидкой краски на бумаге (10%…25%).

2. **FM-растрирование (Stochastic / Frequency Modulated):**
   * Псевдослучайное распределение микроточек (Blue Noise / Error Diffusion Algorithm).

### 3.3. Моделирование дефектов сканирования и физического носителя
К сгенерированному растру последовательно применяются случайные аугментации:
* **Текстура подложки (Substrate Grain):** Наложение волокон бумаги и шероховатости через blend-режимы (Overlay/SoftLight).
* **Геометрические искажения и микро-поворот:** Поворот изображения на мелкие углы (0.1°…2.0°) для сбоя идеального выравнивания сетки и провоцирования вторичного муара при сканировании.
* **Оптический расфокус (Lens Defocus / Blur):** Gaussian Blur (σ ∈ [0.3, 1.2]) для имитации неидеальной фокусировки сканера.
* **Шумы сенсора:** Смесь шума Гаусса и Пуассона (фотонный шум CIS/CCD матрицы).
* **Пыль и царапины:** Генерация случайных тонких белых/черных нитей и микроточек.

---

## 4. Фаза 2: Архитектура нейросети AI Fourier-UNet

Обычная UNet плохо справляется с дескринингом, так как локальные свертки 3×3 не видят глобальную периодичность растра. В основу архитектуры положен **Fast Fourier Convolution (FFC)** блок.

```
                  ┌─────────────────────────────────────────┐
                  │              Input Patch                │
                  └────────────────────┬────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌──────────────────────┐              ┌──────────────────────┐
        │    Spatial Branch    │              │   Spectral Branch    │
        │ (Local Convolutions) │              │  (2D FFT Processing) │
        └───────────┬──────────┘              └───────────┬──────────┘
                    │                                     │
                    │   ┌─────────────────────────────┐   │
                    └──►│ 2D rFFT                     │◄──┘
                        │ Complex Conv (Real + Imag)  │
                        │ Inverse 2D rFFT             │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │ Dynamic Feature Fusion      │
                        └──────────────┬──────────────┘
                                       │
                                       ▼
                        ┌─────────────────────────────┐
                        │    Predicted Residual Mask  │
                        │    (Raster & Noise Removal) │
                        └─────────────────────────────┘
```

### 4.1. Конструкция FFC-блока (PyTorch)
Сеть обучается предсказывать **остаточный шум (Residual Prediction)**: Y_hat = X - Net(X), где X — зашумленный скан, а Net(X) выделяет сетку растра и шум.

```python
import torch
import torch.nn as nn

class FourierSpectralBlock(nn.Module):
    """Блок обработки глобальных частотных характеристик в Fourier-домене."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_real = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.conv_imag = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ffted = torch.fft.rfft2(x, norm='ortho')
        
        real = ffted.real
        imag = ffted.imag
        
        out_real = self.conv_real(real) - self.conv_imag(imag)
        out_imag = self.conv_real(imag) + self.conv_imag(real)
        
        out_fft = torch.complex(out_real, out_imag)
        output = torch.fft.irfft2(out_fft, s=(H, W), norm='ortho')
        return output

class FourierUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.ffc_block1 = FourierSpectralBlock(64, 64)
        self.spatial_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.outc = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        feat = torch.nn.functional.gelu(self.inc(x))
        s_feat = self.spatial_conv(feat)
        f_feat = self.ffc_block1(feat)
        
        combined = torch.cat([s_feat, f_feat], dim=1)
        residual = self.outc(combined)
        return torch.clamp(x - residual, 0.0, 1.0)
```

### 4.2. Комплексная функция потерь (Composite Loss Function)
Для обучения используется взвешенная комбинация четырех лоссов:

Loss_total = L1 + λ1 * Loss_MS-SSIM + λ2 * Loss_Fourier_Mag + λ3 * Loss_LPIPS

1. **Spatial Loss (L1):** Точность пиксельной реконструкции цвета (L1-norm).
2. **Structural Loss (MS-SSIM):** Сохранение контраста и четких краев объектов/букв.
3. **Spectral Magnitude Loss (Fourier_Mag):** Вычисляется как L1-расстояние между амплитудными спектрами предсказанного и идеального изображений.
4. **Perceptual Loss (LPIPS):** Сохранение естественности изображения по восприятию человеком.

---

## 5. Фаза 3: Конвертация и оптимизация ONNX

### 5.1. Экспорт PyTorch → ONNX
Модель экспортируется с динамическими размерами осей для возможности обработки фрагментов произвольного размера.

```python
import torch

def export_fourier_unet():
    model = FourierUNet().eval()
    dummy_input = torch.randn(1, 3, 512, 512)

    torch.onnx.export(
        model,
        dummy_input,
        "fourier_descreen_unet.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print("ONNX модель успешно экспортирована.")

if __name__ == "__main__":
    export_fourier_unet()
```

### 5.2. Квантование и оптимизация графика
* **FP16 Optimization:** Конвертация весов в FP16 для ускорения на GPU через TensorRT / DirectML.
* **Graph Optimization:** Запуск `onnxruntime.transformers.optimizer` для слияния блоков Conv+GELU.

---

## 6. Фаза 4: Backend Engine на Rust + Tauri v2

Бэкенд на Rust отвечает за эффективный ввод-вывод TIFF/PNG сканов гигапиксельного размера, разбиение на тайлы, параллельный инференс через ONNX Runtime и сборку результата без артефактов на границах.

### 6.1. Зависимости `Cargo.toml`
```toml
[package]
name = "descreen-studio-core"
version = "0.3.0"
edition = "2021"

[dependencies]
tauri = { version = "2.0", features = [] }
ort = { version = "2.0.0-rc.9", features = ["directml", "coreml", "cuda"] }
ndarray = "0.15"
image = "0.25"
rayon = "1.10"
rustfft = "6.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### 6.2. Алгоритм плитной обработки (Tiled Overlap-Add with Hann Window)
Для того чтобы на стыках тайлов 512×512 не возникало видимых швов, применяются **окна Ханна (Hann Windowing)** при сшивании перекрывающихся областей (Overlap = 64px).

```rust
// src-tauri/src/inference.rs

use ort::{inputs, ExecutionProviderDispatch, GraphOptimizationLevel, Session};
use ndarray::{Array4, ArrayViewMut4};
use rayon::prelude::*;

pub struct InferenceEngine {
    session: Session,
}

impl InferenceEngine {
    pub fn new(model_bytes: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers([
                #[cfg(target_os = "windows")]
                ort::DirectMLExecutionProvider::default().build(),
                #[cfg(target_os = "macos")]
                ort::CoreMLExecutionProvider::default().build(),
            ])?
            .commit_from_memory(model_bytes)?;

        Ok(Self { session })
    }

    pub fn process_tile(&self, input_patch: Array4<f32>) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
        let outputs = self.session.run(inputs!["input" => input_patch.view()]?)?;
        let output_tensor = outputs["output"].extract_tensor::<f32>()?;
        
        let view = output_tensor.view();
        Ok(view.to_shared().into_dimensionality::<ndarray::Ix4>()?)
    }
}
```

---

## 7. Фаза 5: Frontend Interface (Svelte 5 + Tailwind CSS v4)

Интерфейс предоставляет рабочий стол для реставраторов и операторов сканирования с моментальным просмотром частотного спектра изображения.

### 7.1. Svelte 5 Компонент управления и взаимодействия (IPC)
```svelte
<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';

  let aiBlend = $state(0.85);
  let isProcessing = $state(false);
  let imagePath = $state<string | null>(null);

  async function runDescreenProcess() {
    if (!imagePath) return;
    isProcessing = true;
    
    try {
      const result = await invoke('run_fourier_descreen', { 
        path: imagePath, 
        blend: aiBlend 
      });
      console.log('Обработка завершена:', result);
    } catch (err) {
      console.error('Ошибка инференса:', err);
    } finally {
      isProcessing = false;
    }
  }
</script>

<main class="flex h-screen bg-neutral-900 text-neutral-100 font-sans">
  <section class="flex-1 flex flex-col border-r border-neutral-800">
    <header class="h-12 border-b border-neutral-800 flex items-center px-4 justify-between">
      <span class="text-xs font-semibold uppercase tracking-wider text-neutral-400">Preview Viewport</span>
    </header>
    <div class="flex-1 bg-black flex items-center justify-center relative overflow-hidden">
      <!-- HTML5 Canvas Viewport -->
    </div>
  </section>

  <aside class="w-80 flex flex-col bg-neutral-950 p-4 gap-6">
    <div>
      <h3 class="text-sm font-medium mb-3 text-neutral-300">Частотный спектр (2D FFT)</h3>
      <div class="aspect-square bg-neutral-900 rounded-lg border border-neutral-800 flex items-center justify-center">
        <!-- FFT Canvas Visualization -->
      </div>
    </div>

    <div class="flex flex-col gap-4">
      <h3 class="text-sm font-medium text-neutral-300">Параметры нейросети</h3>
      
      <label class="flex flex-col gap-1.5 text-xs text-neutral-400">
        <span>Сила подавления растра (AI Blend): {Math.round(aiBlend * 100)}%</span>
        <input type="range" min="0" max="1" step="0.05" bind:value={aiBlend} class="accent-indigo-500"/>
      </label>

      <button 
        onclick={runDescreenProcess}
        disabled={isProcessing}
        class="mt-4 py-2.5 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-neutral-800 rounded-md font-medium text-sm transition-colors cursor-pointer flex items-center justify-center gap-2">
        {#if isProcessing}
          <span>Обработка на GPU...</span>
        {:else}
          <span>Запустить дескрининг</span>
        {/if}
      </button>
    </div>
  </aside>
</main>
```

---

## 8. План поэтапной реализации и критерии приемки

| Этап | Задача | Результат / Deliverable | Срок |
| :--- | :--- | :--- | :--- |
| **Этап 1** | **Dataset Pipeline** | Python-скрипт синтеза пара растр/оригинал с AM/FM растром, TVI и шумами (10k+ патчей). | 1.5 нед. |
| **Этап 2** | **Model & Training** | Реализация Fourier-UNet на PyTorch. Достижение показателей: **PSNR > 34 dB**, **SSIM > 0.95**. | 2 нед. |
| **Этап 3** | **ONNX Export & Optimization** | Готовый `fourier_descreen_unet.onnx` с динамическими осями и FP16 квантованием. | 3 дня |
| **Этап 4** | **Rust/Tauri Backend** | Интеграция `ort` с DirectML/CoreML, реализация Tiling & Overlap-Add алгоритма. | 1.5 нед. |
| **Этап 5** | **Frontend & UI/UX** | Svelte 5 + Tailwind UI с просмотром 2D FFT спектра и режима split-screen сравнения. | 1 нед. |
| **Этап 6** | **Release & Packaging** | Автоматическая сборка через GitHub Actions: `.exe` (Win 10/11) и `.dmg` (macOS). | 3 дня |

---

## 9. Показатели успешности проекта (Acceptance Criteria)

1. **Качество обработки:** Полное удаление типографского растра (AM 85–175 LPI и FM) без размытия текста шрифтом мелкого кегля (от 4 pt) и без появления вторичного муара.
2. **Скорость работы:** Инференс скана формата А4 (300 DPI, ~3500×2500 px) занимает **менее 2.5 секунд** на дискретной видеокарте и **менее 6 секунд** на встроенной графике Intel/AMD через DirectML.
3. **Портативность:** Итоговый размер сборки приложения составляет **менее 70 МБ**, не требует установки Python, CUDA SDK или сторонних зависимостей. Запуск происходит из одной папки/файла `.exe` / `.dmg`.