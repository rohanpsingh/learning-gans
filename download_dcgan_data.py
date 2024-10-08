#!/usr/bin/env python3
import hashlib
import os

import time
from typing import Optional
from urllib.request import urlopen, Request
from pathlib import Path
from zipfile import ZipFile

REPO_BASE_DIR = Path(__file__).absolute().parent
DATA_DIR = REPO_BASE_DIR / "data"

def size_fmt(nbytes: int) -> str:
    """Returns a formatted file size string"""
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    return str(nbytes) + " bytes"


def download_url_to_file(url: str,
                         dst: Optional[str] = None,
                         prefix: Optional[Path] = None,
                         sha256: Optional[str] = None) -> Path:
    dst = dst if dst is not None else Path(url).name
    dst = dst if prefix is None else str(prefix / dst)
    if Path(dst).exists():
        print(f"Skip downloading {url} as {dst} already exists")
        return Path(dst)
    file_size = None
    u = urlopen(Request(url, headers={"User-Agent": "tutorials.downloader"}))
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])
    sha256_sum = hashlib.sha256()

    downloaded_size = 0  # Initialize the downloaded size tracker
    start_time = time.time()
    with open(dst, "wb") as f:
        while True:
            buffer = u.read(32768)
            if len(buffer) == 0:
                break
            downloaded_size += len(buffer)  # Update the downloaded size
            sha256_sum.update(buffer)
            f.write(buffer)
            # Print the download progress
            if file_size:
                progress = (downloaded_size / file_size) * 100
                elapsed_time = time.time() - start_time
                speed = downloaded_size / elapsed_time  # bytes per second
                eta = (file_size - downloaded_size) / speed if speed > 0 else 0  # remaining time in seconds
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))  # format ETA as HH:MM:SS
                print(f"\rDownloading... {progress:.2f}% completed, ETA: {eta_str}", end="")
    print()

    digest = sha256_sum.hexdigest()
    if sha256 is not None and sha256 != digest:
        Path(dst).unlink()
        raise RuntimeError(f"Downloaded {url} has unexpected sha256sum {digest} should be {sha256}")
    print(f"Downloaded {url} sha256sum={digest} size={size_fmt(file_size)}")
    return Path(dst)


def unzip(archive: Path, tgt_dir: Path) -> None:
    with ZipFile(str(archive), "r") as zip_ref:
        zip_ref.extractall(str(tgt_dir))

def download_dcgan_data() -> None:
    # Download dataset for beginner_source/dcgan_faces_tutorial.py
    z = download_url_to_file("https://s3.amazonaws.com/pytorch-tutorial-assets/img_align_celeba.zip",
                             prefix=DATA_DIR,
                             sha256="46fb89443c578308acf364d7d379fe1b9efb793042c0af734b6112e4fd3a8c74",
                             )
    unzip(z, DATA_DIR / "celeba")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    download_dcgan_data()

if __name__ == "__main__":
    main()
