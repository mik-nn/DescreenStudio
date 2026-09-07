<!-- STREAMING_CHUNK:Writing header and badges... -->
# Descreening Studio Pro v3.7 🌊✨

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Active-emerald.svg)]()
[![Platform: Web Browser](https://img.shields.io/badge/Platform-Client--side%20HTML5-blue.svg)]()

> **English** | **[Русский](#-русская-версия)**

**Descreening Studio Pro** is an advanced, purely client-side browser application designed for professional halftone moiré pattern removal from scanned images, print materials, magazines, and engravings. Utilizing **2D Fast Fourier Transform (FFT)**, **Butterworth Notch Filtering**, and **Edge-Preserving Guided Denoising**, it allows users to effectively restore scanned artwork to crisp digital clarity without external backend servers.

---

## 🚀 Key Features

- ⚛️ **2D FFT & Inverse FFT Processing**: Filter moiré patterns directly in the frequency domain with high-precision complex number calculations.
- 🎯 **Automated Symmetry Notch Detection**: Smart statistical outlier algorithm ($\sigma$-thresholding) that detects periodic halftone dots and automatically applies Butterworth notch filters.
- 🎨 **Interactive Spectrum Canvas**: Erase frequency harmonics manually with custom brush tools in real-time.
- 🔍 **Real-Time 1:1 Split Comparator**: Interactive split view slider to evaluate "Before vs. After" results at full resolution.
- 🖼️ **Native TIFF & Ultra-High Res Export**: Native browser decoding for `.tif`/`.tiff` files via UTIF.js, plus tile-based overlap-add 2D Hann window processing for full-resolution exports without memory crashes.
- 🎛️ **Custom Presets & Local Storage**: Save custom parameter profiles for newspaper, magazine, or fine-engraving scans with instant `localStorage` saving.
- 🌐 **Multilingual Interface**: Full support for both **English** and **Russian** languages.
- 🔒 **100% Client-Side Privacy**: All processing runs locally inside your Web Browser—your images are never uploaded to any server.

---

## 🛠️ Built With

- **HTML5 & Vanilla JavaScript** (Zero build tools required)
- **Tailwind CSS** (Styling & Responsive UI)
- **UTIF.js** (TIFF File Reader)
- **FontAwesome 6** (Icons)

---

## ⚡ Quick Start / Local Setup

1. **Clone or Download** the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/descreening-studio-pro.git
   ```
2. **Open the App**: Simply double-click `index.html` (or `descreen_tool.html`) to launch it in any modern web browser (Chrome, Firefox, Edge, Safari). No `npm install` or local server setup required!

---

## 📖 Usage Guide

1. **Open Scan**: Click the **Open Scan** button to load a scanned file (`.png`, `.jpg`, `.tif`, `.tiff`) or drag and drop a file onto the interface.
2. **Auto-Detect Peaks**: Click **Detect** under *Halftone Peak Detection* to automatically locate and filter frequency spikes caused by halftone printing grids.
3. **Manual Retouching**: Use your mouse cursor on the **2D FFT Amplitude Spectrum** canvas on the left to manually paint over lingering moiré harmonics.
4. **Fine-Tune**:
   - Adjust **Notch Radius ($D_0$)** and **Butterworth Order ($n$)**.
   - Apply **Guided Denoise** & **Unsharp Mask** for crisp, smooth edges.
5. **Export**: Click **Save Full File** to render and download the cleaned high-resolution image (`_DS.png`).

---

<br>

---

# 🇷🇺 Русская Версия

# Descreening Studio Pro v3.7 🌊✨

**Descreening Studio Pro** — это мощное веб-приложение для профессионального устранения полиграфического растра (муара) со сканов журналов, газет, книг и гравюр. Используя **Двумерное Быстрое Преобразование Фурье (2D БПФ)**, **Фильтры Баттерворта** и **Guided Denoise**, приложение эффективно восстанавливает четкость изображений прямо в вашем браузере.

---

## 🚀 Основные Возможности

- ⚛️ **Фильтрация в Спектре 2D БПФ**: Подавление паттернов растра напрямую в частотной области.
- 🎯 **Автоматический Поиск Пиков Растра**: Умный алгоритм выявления периодических структур ($\sigma$-порог) с авто-наложением вырезающих режекторных фильтров Баттерворта.
- 🎨 **Интерактивная Ретушь Спектра**: Ручное стирание лишних гармоник кистью прямо на холсте спектра в реальном времени.
- 🔍 **Сплит-Сравнение 1:1**: Удобная шторка для мгновенной оценки результата "До / После" в масштабе 1:1.
- 🖼️ **Поддержка TIFF и Экспорт Высокого Разрешения**: Декодирование файлов `.tif`/`.tiff` прямо в браузере и плиточная тайловая обработка больших сканов методом перекрытия (Overlap-Add с окном Ханна).
- 🎛️ **Пресеты и Сохранение Настроек**: Встроенные профили (журналы, газеты, гравюры) и возможность сохранять свои настройки в `localStorage`.
- 🌐 **Двуязычный Интерфейс**: Полная поддержка **Русского** и **Английского** языков.
- 🔒 **Полная Конфиденциальность**: Все вычислительные процессы происходят на стороне клиента — ваши файлы не отправляются на сторонние серверы.

---

## ⚡ Быстрый Запуск

1. **Склонируйте репозиторий**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/descreening-studio-pro.git
   ```
2. **Откройте файл**: Просто откройте `index.html` (или `descreen_tool.html`) в любом современном браузере. Установка зависимостей или запуск веб-сервера не требуются!

---

## 📜 Лицензия / License

Distributed under the MIT License. See `LICENSE` for more information.