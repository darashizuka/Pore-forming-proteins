#link to file : https://drive.google.com/file/d/1ASOTaVCfUDS9DaWZ3yf7Q_CswJCw7Yha/view?usp=sharing
import os
import zipfile
import subprocess


def download_and_extract(file_id, destination="."):
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call(["pip", "install", "gdown"])
        import gdown

    zip_path = os.path.join(destination, "Data.zip")

    print("Downloading Data.zip from Google Drive...")
    gdown.download(id=file_id, output=zip_path, quiet=False)

    print("Extracting Data.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

    print("Cleaning up zip file...")
    os.remove(zip_path)

    print("Data folder is ready.")

if __name__ == "__main__":
    # Replace this with your actual file ID
    FILE_ID = "1ASOTaVCfUDS9DaWZ3yf7Q_CswJCw7Yha"
    download_and_extract(FILE_ID)
