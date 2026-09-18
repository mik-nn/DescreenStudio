"""
Конвертер модели PyTorch -> Safetensors для Candle
Запускать локально на машине с установленной моделью
"""
import torch
import safetensors.torch
import sys
from pathlib import Path

def convert_model(pt_path: str, output_path: str):
    """Конвертирует .pt/.pth файл в .safetensors"""
    print(f"Загрузка модели из {pt_path}...")
    
    # Загружаем веса
    state_dict = torch.load(pt_path, map_location='cpu')
    
    # Если это полный чекпоинт с optimizer состоянием, извлекаем только weights
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        print("Извлечены веса модели из чекпоинта")
    
    # Проверяем наличие EMA весов
    if 'ema_state_dict' in state_dict:
        state_dict = state_dict['ema_state_dict']
        print("Использованы EMA веса")
    
    # Сохраняем в safetensors формате
    print(f"Сохранение в {output_path}...")
    safetensors.torch.save_file(state_dict, output_path)
    
    # Информация о модели
    print("\n=== Информация о модели ===")
    total_params = 0
    for name, tensor in state_dict.items():
        params = tensor.numel()
        total_params += params
        print(f"{name}: {tensor.shape} ({params:,} params)")
    
    print(f"\nВсего параметров: {total_params:,}")
    print(f"Размер файла: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    return state_dict

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python convert_to_safetensors.py <path_to_pt_file> [output_path]")
        print("Пример: python convert_to_safetensors.py F:\\DescreenStudioPro\\ml\\train\\_run3\\best.pt")
        sys.exit(1)
    
    pt_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else pt_path.replace('.pt', '.safetensors').replace('.pth', '.safetensors')
    
    try:
        convert_model(pt_path, output_path)
        print(f"\n✓ Конвертация успешна: {output_path}")
    except Exception as e:
        print(f"\n✗ Ошибка конвертации: {e}")
        sys.exit(1)
