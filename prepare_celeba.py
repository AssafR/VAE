import random
import pickle
import zipfile
import os
import sys
from pathlib import Path
import numpy as np

# Modern image processing libraries
from PIL import Image
import imageio.v2 as imageio

import tqdm

# --- Fixed download section (replaces: from dlutils import download; download.from_google_drive(...)) ---
def download_celeba_zip(dst_path: str = "img_align_celeba.zip") -> str:
    """Download the CelebA images zip from Google Drive using gdown.
    Returns the path to the downloaded file.
    """
    file_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    url = f"https://drive.google.com/uc?id={file_id}"
    dst = Path(dst_path)

    if dst.exists() and dst.stat().st_size > 0:
        print(f"[download] Found existing file: {dst} ({dst.stat().st_size} bytes)")
        return str(dst)

    try:
        import gdown
    except ImportError:
        # Install gdown if missing (standard Python environment)
        import subprocess, sys
        print("[download] Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    print(f"[download] Downloading CelebA to {dst} ...")
    gdown.download(url, str(dst), quiet=False)
    if not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError("Download failed or produced an empty file.")
    return str(dst)

# Trigger the download (idempotent)
# download_celeba_zip()


ZIP_PATH = Path("img_align_celeba.zip")

if not ZIP_PATH.exists():
    print("[ERROR] CelebA dataset not found.")
    print("Please download 'img_align_celeba.zip' from the official site:")
    print("  https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html")
    print("Place it in this folder and re-run the script.")
    sys.exit(1)

print(f"[INFO] Found CelebA zip: {ZIP_PATH}")

corrupted = [
    '195995.jpg',
    '131065.jpg',
    '118355.jpg',
    '080480.jpg',
    '039459.jpg',
    '153323.jpg',
    '011793.jpg',
    '156817.jpg',
    '121050.jpg',
    '198603.jpg',
    '041897.jpg',
    '131899.jpg',
    '048286.jpg',
    '179577.jpg',
    '024184.jpg',
    '016530.jpg',
]


target_filename = "img_align_celeba.zip"
# check if the file exists and is at least 1000MB
if os.path.exists(target_filename) and os.path.getsize(target_filename) < 1000 * 1024 * 1024:
    print(f"File {target_filename} exists and is less than 1000MB, deleting...")
    os.remove(target_filename)
    try:
        # Download the CelebA dataset from Google Drive
        url = 'https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM'
        gdown.download(url, "img_align_celeba.zip", quiet=False)
    except Exception as e:
        print(f"gdown failed: {e}")
        print("Trying alternative gdown method...")
        try:
            # Try with different gdown parameters
            gdown.download(url, "img_align_celeba.zip", quiet=False, fuzzy=True)
        except Exception as e2:
            print(f"Alternative gdown also failed: {e2}")
            print("Trying direct Google Drive download...")
            import requests
            import os
            
            # Use the direct download URL format
            # url = "https://drive.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM"
            filename = "img_align_celeba.zip"
            
            # Check if file exists and get its size for resume
            resume_header = {}
            if os.path.exists(filename):
                file_size = os.path.getsize(filename)
                resume_header = {'Range': f'bytes={file_size}-'}
                print(f"Resuming download from {file_size} bytes...")
            
            # Add user agent to avoid some Google Drive restrictions
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            headers.update(resume_header)
            
            response = requests.get(url, headers=headers, stream=True)
            
            # Check if we got HTML instead of the file
            if response.text.startswith('<!DOCTYPE html>'):
                print("Received HTML instead of file. Google Drive security warning detected.")
                print(f"Please download manually from: {url}")
                print("Or try running the script again later.")
                exit(1)
            
            # Open file in append mode if resuming, write mode if new
            mode = 'ab' if os.path.exists(filename) and resume_header else 'wb'
            with open(filename, mode) as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"Download completed: {filename}")


def center_crop(x, crop_h=128, crop_w=None, resize_w=128):
    # crop the images to [crop_h,crop_w,3] then resize to [resize_h,resize_w,3]
    if crop_w is None:
        crop_w = crop_h # the width and height after cropped
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.)) + 15
    i = int(round((w - crop_w)/2.))
    cropped = x[j:j+crop_h, i:i+crop_w]

    # Convert numpy array → Pillow Image → resize → numpy array
    return np.array(Image.fromarray(cropped).resize((resize_w, resize_w), Image.Resampling.BILINEAR))

archive = zipfile.ZipFile('img_align_celeba.zip', 'r')

names = archive.namelist()

names = [x for x in names if x[-4:] == '.jpg']

count = len(names)
print("Count: %d" % count)

names = [x for x in names if x[-10:] not in corrupted]

folds = 5

random.shuffle(names)

images = {}

count = len(names)
print("Count: %d" % count)
count_per_fold = count // folds

i = 0
im = 0
for x in tqdm.tqdm(names):
    imgfile = archive.open(x)
    image = imageio.imread(imgfile)
    image = center_crop(image)
    images[x] = image
    im += 1

    if im == count_per_fold:
        output = open('data_fold_%d.pkl' % i, 'wb')
        pickle.dump(list(images.values()), output)
        output.close()
        i += 1
        im = 0
        images.clear()
