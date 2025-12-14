import os
from pathlib import Path
import urllib.request
import zipfile

def download_sample_logs(url, logs_filename:str="sample_logs"):
    sample_dir = Path(logs_filename)
    if sample_dir.exists():
        print("Sample logs already downloaded.")
        return sample_dir

    zip_path = "sample_logs.zip"

    _ = urlib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()

    os.remove(zip_path)
    print("Sample logs downloaded successfully.")
    return sample_dir
