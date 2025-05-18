import os
import tarfile
import requests
import pandas as pd
from typing import Tuple

def download_dataset(url: str, save_path: str) -> None:
    """Download and extract dataset"""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    tar_path = os.path.join(save_path, "data.tar")
    
    if not os.path.exists(tar_path):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {str(e)}")
    
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(save_path)
    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {str(e)}")

def get_dataset_paths(language: str, data_dir: str) -> Tuple[str, str, str]:
    """Get paths to train/dev/test files"""
    base_path = os.path.join(data_dir, f"dakshina_dataset_v1.0/{language}/lexicons")
    return (
        os.path.join(base_path, f"{language}.translit.sampled.train.tsv"),
        os.path.join(base_path, f"{language}.translit.sampled.dev.tsv"),
        os.path.join(base_path, f"{language}.translit.sampled.test.tsv")
    )

def load_tsv(file_path: str) -> pd.DataFrame:
    """Load TSV file into DataFrame"""
    try:
        return pd.read_csv(file_path, sep='\t', header=None, names=['target', 'input'])
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {file_path}")