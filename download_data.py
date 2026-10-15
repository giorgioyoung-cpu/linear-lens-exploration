from pathlib import Path
from datasets import load_dataset

data_dir = Path.home() / "jspace-data"
data_dir.mkdir(parents=True, exist_ok=True)

dataset = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    cache_dir=str(data_dir / "dataset-cache"),
)

save_path = data_dir / "wikitext-2"
dataset.save_to_disk(str(save_path))

print(dataset)
print("Saved to:", save_path)