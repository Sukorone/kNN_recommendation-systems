#!/usr/bin/env python3
"""
Скрипт для проверки установки всех необходимых модулей
"""

def check_modules():
    """Проверяет наличие всех необходимых модулей"""
    required_modules = ['numpy', 'pandas']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"[OK] {module} - установлен")
        except ImportError:
            print(f"[ERROR] {module} - НЕ установлен")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n[ERROR] Отсутствуют модули: {', '.join(missing_modules)}")
        print("Установите их командой:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    else:
        print("\n[OK] Все модули установлены!")
        return True

if __name__ == "__main__":
    check_modules()
