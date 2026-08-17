

import shutil
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_NAME = "Parveshiiii/AI-vs-Real"
OUTPUT_DIR = Path("data/parveshiiii_ai_vs_real")

VAL_FRACTION = 0.1 
SEED = 42

IMAGE_COLUMN_CANDIDATES = ("image", "img")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "target", "class", "binary_label")


def detect_columns(ds):
    """Определяет реальные имена колонок картинки и метки из features датасета,
    а не гадает — если нужной колонки нет, падаем с понятной ошибкой вместо
    того, чтобы молча всё промаркировать одним классом."""
    columns = set(ds.column_names)

    image_col = next((c for c in IMAGE_COLUMN_CANDIDATES if c in columns), None)
    label_col = next((c for c in LABEL_COLUMN_CANDIDATES if c in columns), None)

    if image_col is None or label_col is None:
        raise RuntimeError(
            f"Не смог определить колонки картинки/метки автоматически.\n"
            f"Реальные колонки датасета: {sorted(columns)}\n"
            f"Типы колонок (ds.features): {dict(ds.features)}\n"
            f"Пример первой строки (без содержимого картинки): "
            f"{ {k: v for k, v in ds[0].items() if k != image_col} }\n"
            f"Добавь нужное имя колонки в IMAGE_COLUMN_CANDIDATES / LABEL_COLUMN_CANDIDATES "
            f"в начале скрипта и запусти заново."
        )

    return image_col, label_col


def build_label_resolver(ds, label_col: str):
    """Строит функцию int/str-метка -> 'real'/'fake', используя официальную
    схему датасета (ClassLabel.names), а не наши догадки о том, что 0/1
    означают."""
    feature = ds.features[label_col]

    if hasattr(feature, "int2str"):
        names = [n.lower() for n in feature.names]
        print(f"ℹ️  Колонка меток '{label_col}' — ClassLabel, классы: {feature.names}")

        def resolve(value) -> str:
            name = feature.int2str(value).lower() if isinstance(value, int) else str(value).lower()
            return "real" if "real" in name else "fake"

        return resolve

    print(f"ℹ️  Колонка меток '{label_col}' — без схемы ClassLabel, определяю эвристикой по значению.")

    def resolve(value) -> str:
        if isinstance(value, str):
            return "real" if "real" in value.lower() else "fake"
        return "real" if value == 1 else "fake"

    return resolve


def save_split(ds, local_split: str, counts: dict, image_col: str, label_col: str, resolve_label):
    print(f"⏳ Раскладываю {len(ds)} изображений -> {OUTPUT_DIR / local_split}/")

    for item in tqdm(ds, desc=f"-> {local_split}"):
        img: Optional[Image.Image] = item.get(image_col)
        if img is None:
            continue

        cls_name = resolve_label(item.get(label_col))
        idx = counts[local_split][cls_name]
        save_path = OUTPUT_DIR / local_split / cls_name / f"{cls_name}_{idx:05d}.jpg"

        try:
            img.convert("RGB").save(save_path, "JPEG")
        except Exception as e:
            print(f"⚠️  Не удалось сохранить изображение {idx} ({cls_name}, {local_split}): {e}")
            continue

        counts[local_split][cls_name] += 1


def download_and_distribute():
    if OUTPUT_DIR.exists():
        print(f"⚠️  Папка {OUTPUT_DIR} уже существует — удаляю и создаю заново.")
        shutil.rmtree(OUTPUT_DIR)

    for local_split in ("train", "val"):
        for cls in ("real", "fake"):
            (OUTPUT_DIR / local_split / cls).mkdir(parents=True, exist_ok=True)

    print(f"📦 Скачиваем {DATASET_NAME} (появится стандартный прогресс-бар Hugging Face)...")
    ds_dict = load_dataset(DATASET_NAME, verification_mode="no_checks")

    available_splits = list(ds_dict.keys())
    print(f"✅ Датасет загружен. Доступные сплиты на HF: {available_splits}")

    reference_ds = ds_dict["train"]
    image_col, label_col = detect_columns(reference_ds)
    resolve_label = build_label_resolver(reference_ds, label_col)

    counts = {"train": {"real": 0, "fake": 0}, "val": {"real": 0, "fake": 0}}

    if "test" in ds_dict:
        save_split(ds_dict["train"], "train", counts, image_col, label_col, resolve_label)
        save_split(ds_dict["test"], "val", counts, image_col, label_col, resolve_label)
    else:
        print(f"ℹ️  Отдельного test-сплита нет, делю 'train' сам: {1 - VAL_FRACTION:.0%} train / {VAL_FRACTION:.0%} val")
        full = ds_dict["train"].shuffle(seed=SEED)
        val_size = int(len(full) * VAL_FRACTION)

        val_ds = full.select(range(val_size))
        train_ds = full.select(range(val_size, len(full)))

        save_split(train_ds, "train", counts, image_col, label_col, resolve_label)
        save_split(val_ds, "val", counts, image_col, label_col, resolve_label)

    print("\n📊 Итоговое распределение:")
    for local_split, cls_counts in counts.items():
        total = sum(cls_counts.values())
        print(f"  {local_split}: real={cls_counts['real']}, fake={cls_counts['fake']} (всего {total})")

    if any(cls_counts["real"] == 0 or cls_counts["fake"] == 0 for cls_counts in counts.values()):
        print(
            "\n⚠️  Один из классов пуст — что-то по-прежнему определяется неверно. "
            "Проверь вывод detect_columns/build_label_resolver выше и, если нужно, "
            "пришли реальные имена колонок датасета."
        )
    else:
        total_real = sum(c["real"] for c in counts.values())
        total_fake = sum(c["fake"] for c in counts.values())
        ratio = total_real / (total_real + total_fake)
        print(f"\nℹ️  Баланс классов по всему датасету: real={ratio:.1%}, fake={1 - ratio:.1%}")
        if not (0.4 <= ratio <= 0.6):
            print(
                "⚠️  Датасет заявлен автором как сбалансированный (~50/50), а получившийся "
                "баланс сильно отклоняется от этого. Это может значить, что классы "
                "'real' и 'fake' перепутаны местами (например, реальный маппинг "
                "binary_label наоборот: 0=Real, 1=AI). Проверь вручную несколько "
                "картинок из data/parveshiiii_ai_vs_real/train/real и .../fake глазами, "
                "прежде чем запускать обучение на этих данных."
            )

    print(f"\n✅ Готово. Данные лежат в {OUTPUT_DIR}/")


if __name__ == "__main__":
    download_and_distribute()