# Candle Benchmark Guide

## Обзор

Бенчмарк Candle для Descreen Studio Pro позволяет тестировать производительность модели FourierUNet на различных устройствах (CPU, CUDA, Metal) после конвертации из PyTorch в формат Safetensors.

## Установка

### 1. Rust и зависимости

```bash
# Установка Rust (если не установлен)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# После установки перезапустите терминал или выполните:
source $HOME/.cargo/env

# Проверка установки
rustc --version
cargo --version
```

### 2. Конвертация модели

Сначала сконвертируйте веса модели из PyTorch (.pt) в Safetensors (.safetensors):

```bash
cd /workspace/ml/benchmark

# Установка зависимостей Python (если нужны)
pip install safetensors torch

# Конвертация модели
python convert_to_safetensors.py F:\\DescreenStudioPro\\ml\\train\\_run3\\best.pt
```

Файл будет сохранен как `best.safetensors` в той же директории.

### 3. Сборка бенчмарка

```bash
cd /workspace/src-tauri

# Debug сборка (быстрее)
cargo build --bin candle_bench

# Release сборка (оптимизирована, рекомендуется для бенчмарков)
cargo build --release --bin candle_bench
```

## Запуск

### Базовый запуск (CPU)

```bash
# Из директории проекта
cargo run --release --bin candle_bench -- --model path/to/model.safetensors

# Или прямой запуск бинарника
./target/release/candle_bench --model path/to/model.safetensors
```

### С опциями

```bash
cargo run --release --bin candle_bench -- \
  --model best.safetensors \
  --h 2480 \
  --w 3508 \
  --tile 512 \
  --overlap 64 \
  --device cpu \
  --warmup 3 \
  --runs 5 \
  --verify
```

### На GPU (CUDA)

```bash
cargo run --release --bin candle_bench -- \
  --model best.safetensors \
  --device cuda \
  --h 1024 --w 1024 \
  --warmup 5 --runs 10
```

### На Apple Silicon (Metal)

```bash
cargo run --release --bin candle_bench -- \
  --model best.safetensors \
  --device metal
```

## Опции командной строки

| Опция | Описание | По умолчанию |
|-------|----------|--------------|
| `--model <path>` | Путь к .safetensors файлу модели | model.safetensors |
| `--h <pixels>` | Высота тестового изображения | 2480 |
| `--w <pixels>` | Ширина тестового изображения | 3508 |
| `--tile <size>` | Размер тайла в пикселях | 512 |
| `--overlap <pixels>` | Перекрытие между тайлами | 64 |
| `--device <cpu\|cuda\|metal>` | Устройство для инференса | cpu |
| `--warmup <N>` | Количество прогревающих итераций | 3 |
| `--runs <N>` | Количество тестовых прогонов | 5 |
| `--verify` | Проверка корректности вывода | false |
| `--help` | Показать справку | - |

## Интерпретация результатов

Пример вывода:

```
=== Candle Benchmark ===
Устройство: Cpu
Размер изображения: 3508x2480
Тайл: 512px, перекрытие: 64px
Модель: best.safetensors

Загрузка модели из best.safetensors...
Модель загружена успешно
Параметров: ~1.5M

Генерация тестового изображения...
Изображение: Shape([1, 3, 2480, 3508]), размер: 3508x2480

Проверка на одиночном тайле...
Первый прогон (тайл 512x512): 45.23ms
Выходной тензор: Shape([1, 3, 512, 512])
Статистика вывода: mean=0.5123

=== Полный бенчмарк ===
Количество тайлов: 42 (6x7)
Прогрев (3 итераций)...
Прогон 1/5... 1890.45ms всего, 45.01ms/тайл
...

=== Результаты ===
Среднее время: 1875.32ms
Min/Max: 1850.12ms / 1920.45ms
Производительность: 22.38 тайлов/сек
```

### Ключевые метрики

- **ms/тайл**: Время обработки одного тайла (ниже = лучше)
- **тайлов/сек**: Пропускная способность (выше = лучше)
- **Min/Max разброс**: Стабильность производительности

## Архитектура модели

Бенчмарк использует архитектуру FourierUNet с параметрами по умолчанию:

- **in_channels**: 3 (RGB)
- **out_channels**: 3 (RGB)
- **base**: 64 канала
- **levels**: 3 (многосcale UNet)
- **enc_blocks**: [1, 1] (FFC блоки в энкодере)
- **bottleneck_blocks**: 2
- **dec_blocks**: [1, 1] (FFC блоки в декодере)
- **spectral**: [true, true, true] (FFT блоки на всех уровнях)

## Troubleshooting

### Ошибка: "CUDA не доступен"

Убедитесь, что:
1. Установлен CUDA Toolkit
2. GPU NVIDIA поддерживается
3. Candle собран с флагом `cuda`

### Ошибка: "Файл модели не найден"

Сначала сконвертируйте модель:
```bash
python convert_to_safetensors.py path/to/best.pt
```

### Ошибка компиляции: "package candle-core not found"

Обновите зависимости:
```bash
cargo update
cargo build --bin candle_bench
```

## Следующие шаги

1. **Сравнение с ONNX**: Запустите `descreen_bench` с той же моделью в ONNX формате
2. **Тонкая настройка архитектуры**: Измените параметры в `ModelConfig::default()`
3. **Интеграция в приложение**: Используйте код из `candle_bench.rs` как основу для inference engine

## Дополнительные ресурсы

- [Candle Documentation](https://docs.rs/candle-core/)
- [Hugging Face Candle](https://github.com/huggingface/candle)
- [Safetensors Format](https://github.com/huggingface/safetensors)
