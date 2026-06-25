
from datasets import load_dataset
from pathlib import Path
import json

dataset_name = "google-research-datasets/mbpp"
dataset_config = "full"
out_dir = Path("data/raw/mbpp_hf_full")
out_dir.mkdir(parents=True, exist_ok=True)

ds = load_dataset(dataset_name, dataset_config)
print(list(ds.items())[0])
exit()

summary = {
    "dataset_name": dataset_name,
    "dataset_config": dataset_config,
    "splits": {},
}

for split_name, split in ds.items():
    out_path = out_dir / f"{split_name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for item in split:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary["splits"][split_name] = {
        "rows": len(split),
        "columns": list(split.column_names),
        "jsonl": str(out_path),
    }
    print(f"saved {split_name}: {len(split)} rows -> {out_path}")

(out_dir / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print(json.dumps(ds["train"][0], ensure_ascii=False, indent=2))
