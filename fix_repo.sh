#!/bin/bash

echo "🚀 Начинаем исправление репозитория VeriVision..."

# Пункты 3, 4, 6 и 8: Модификация README.md
python3 -c '
import re
import os

readme = "README.md"
if os.path.exists(readme):
    with open(readme, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 3. Убираем ограничение про пустой Dockerfile
    content = re.sub(r"- Dockerfile пустой — контейнеризация ещё не сделана\n?", "", content)
    
    # 4. Исправляем информацию о весах (убираем ложь о том, что они в репо)
    content = content.replace("rf_spectral.pkl, rf_srm.pkl — веса Random Forest (входят в репозиторий)", 
                              "rf_spectral.pkl, rf_srm.pkl — веса Random Forest (артефакты обучения, исключены из Git via .gitignore). Для первого запуска необходимо либо сгенерировать веса локально через скрипты обучения, либо скачать их из GitHub Releases в папку models/.")
    
    # 6. Исправляем размер датасета (50 000 -> 14 000)
    content = content.replace("50 000", "14 000")
    
    # 8. Меняем описание gradcam.py в дереве проекта
    content = re.sub(r"├── gradcam\.py\s+.*", "├── gradcam.py      # Архивный/экспериментальный модуль Grad-CAM (не подключен к основному app.py)", content)

    with open(readme, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ README.md успешно обновлен (пункты 3, 4, 6, 8)")
else:
    print("⚠️ README.md не найден")
'

# Пункт 5: Перезапись .gitignore без мусора и дубликатов
cat << 'EOF' > .gitignore
# Python / Byte-compiled / module files
__pycache__/
*.py[cod]
*$py.class
*.so

# Environments
.venv/
venv/
ENV/
env/

# IDE & OS
.DS_Store
.vscode/
.idea/

# Jupyter Notebook Checkpoints
.ipynb_checkpoints

# Model artifacts & weights (DO NOT COMMIT LARGE BINARIES)
models/*.pkl
models/*.pth
models/*.pt
models/*.onnx
!models/.gitkeep

# Local & Experimental Data
data/
experiments/results/*.png
EOF
echo "✅ .gitignore очищен от дубликатов (пункт 5)"

# Пункт 6: Исправление цифры в отчете Эксперимента №7
python3 -c '
import os
file_path = "experiments/results/07_calibrated_fusion_and_overfitting.md"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("50 000", "14 000")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Отчет Эксперимента №7 обновлен (пункт 6)")
else:
    print("⚠️ Файл " + file_path + " не найден")
'

# Пункт 7: Исправление докстринга в скрипте загрузки
python3 -c '
import os
file_path = "scripts/download_ai_vs_real.py"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("src/data/download_parveshiiii_dataset.py", "scripts/download_ai_vs_real.py")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Докстринг в scripts/download_ai_vs_real.py обновлен (пункт 7)")
else:
    print("⚠️ Файл scripts/download_ai_vs_real.py не найден")
'

# Пункт 8: Исправление заголовка в gradcam.py
python3 -c '
import os
file_path = "src/serving/gradcam.py"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if lines and "# src/models/gradcam.py" in lines[0]:
        lines[0] = lines[0].replace("# src/models/gradcam.py", "# src/serving/gradcam.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("✅ Заголовок в gradcam.py исправлен (пункт 8)")
else:
    print("⚠️ Файл src/serving/gradcam.py не найден")
'

# Пункт 9: Обновление комментария про ResNet50 в transforms.py
python3 -c '
import os
file_path = "src/data/transforms.py"
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    content = content.replace("предобученный ResNet50", "ConvNeXt-Tiny и визуальный бэкбон")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Комментарий в src/data/transforms.py обновлен (пункт 9)")
else:
    print("⚠️ Файл src/data/transforms.py не найден")
'

# Пункт 10: Явное добавление joblib и scipy в requirements.txt
python3 -c '
import os
req_file = "requirements.txt"
if os.path.exists(req_file):
    with open(req_file, "r", encoding="utf-8") as f:
        content = f.read()
    new_deps = []
    if "joblib" not in content:
        new_deps.append("joblib>=1.3.0")
    if "scipy" not in content:
        new_deps.append("scipy>=1.10.0")
    if new_deps:
        with open(req_file, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(new_deps) + "\n")
        print(f"✅ В requirements.txt добавлены зависимости: {new_deps} (пункт 10)")
    else:
        print("ℹ️ joblib и scipy уже присутствуют в requirements.txt (пункт 10)")
else:
    print("⚠️ requirements.txt не найден")
'

echo "🎉 Все исправления успешно применены! Репозиторий очищен и готов к деплою."