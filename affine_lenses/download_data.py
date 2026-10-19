"""Optional first-time download; skip when the model and WikiText are cached."""
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

def main():
    root=Path.home()/'jspace-data'
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct',cache_dir=str(root/'models'),
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    dest=root/'wikitext-2'
    if dest.exists():
        print(f'Keeping existing dataset: {dest}')
    else:
        load_dataset('Salesforce/wikitext','wikitext-2-raw-v1').save_to_disk(str(dest))
        print(f'Saved: {dest}')

if __name__=='__main__': main()
