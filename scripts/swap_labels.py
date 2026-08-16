import shutil
from pathlib import Path

base = Path("data/training_corpus")

# 1. Свап для train
train_fake = base / "train" / "fake"
train_real_web = base / "train" / "real" / "web_hf"
temp_train = base / "train" / "temp_swap"

train_fake.rename(temp_train)
train_real_web.rename(train_fake)
temp_train.rename(train_real_web)

# 2. Свап для val
val_fake = base / "val" / "fake"
val_real_web = base / "val" / "real" / "web_hf"
temp_val = base / "val" / "temp_swap"

val_fake.rename(temp_val)
val_real_web.rename(val_fake)
temp_val.rename(val_real_web)

print("[+] Папки успешно поменяны местами!")
print("    - В train/fake теперь лежат сгенерированные AI")
print("    - В train/real/web_hf теперь лежат реальные фото")