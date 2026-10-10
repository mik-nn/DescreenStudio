import torch
import sys
import os
from safetensors.torch import save_file

def convert_pt_to_safetensors(pt_path, output_path=None):
    print(f"Загрузка модели из {pt_path}...")
    
    try:
        # Загружаем весь чекпоинт. weights_only=False нужен для старых проверок, 
        # но лучше True, если структура простая. Попробуем без strict.
        checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Ошибка загрузки torch.load: {e}")
        # Попытка загрузить с pickle=False если проблема в безопасности (редко для .pt)
        raise e
    
    state_dict = {}
    
    # Разбор структуры чекпоинта
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Найден ключ 'state_dict'.")
        elif 'model' in checkpoint:
            model_obj = checkpoint['model']
            if hasattr(model_obj, 'state_dict'):
                state_dict = model_obj.state_dict()
                print("Извлечен state_dict из объекта 'model'.")
            else:
                # Если 'model' это уже словарь весов, но там есть мусор
                state_dict = {k: v for k, v in checkpoint['model'].items() if isinstance(v, torch.Tensor)}
                print("Обработан словарь 'model' (фильтрация не-тензоров).")
        else:
            # Плоский словарь с мусором (epoch, loss и т.д.)
            print("Прямой словарь чекпоинта. Фильтруем не-тензоры...")
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor):
                    state_dict[k] = v
                else:
                    print(f"  - Пропущено: {k} (тип {type(v)})")
    elif hasattr(checkpoint, 'state_dict'):
        state_dict = checkpoint.state_dict()
    else:
        raise ValueError("Неизвестный формат чекпоинта. Ожидается dict или nn.Module.")

    if not state_dict:
        raise ValueError("Словарь весов пуст!")

    print(f"Подготовлено {len(state_dict)} тензоров для сохранения.")

    if output_path is None:
        base_name = os.path.splitext(pt_path)[0]
        output_path = f"{base_name}.safetensors"

    print(f"Сохранение в {output_path}...")
    # Safetensors требует contiguous тензоры
    final_dict = {k: v.contiguous() for k, v in state_dict.items()}
    
    save_file(final_dict, output_path)
    print("✓ Конвертация успешно завершена!")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python convert_to_safetensors.py <path_to_model.pt> [output_path]")
        sys.exit(1)
    
    pt_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        convert_pt_to_safetensors(pt_file, out_file)
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        sys.exit(1)