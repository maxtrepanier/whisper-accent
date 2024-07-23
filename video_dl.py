import json
import os
import subprocess
from tqdm import tqdm

import numpy as np
import librosa
import soundfile as sf

from typing import List

def select_samples_offsets(duration: int,
                           duration_sample: int = 30,
                           min_separation=25,
                           sr: int = 16e3) -> List[int]:
    """ Select random samples (of duration_sample) from an audio track of given
    duration. The samples are separated by at least min_separation seconds.

    Args:
    duration: total duration of the track (in seconds)
    duration_sample: duration of a single sample (default 30s)
    min_separation: minimal separation between samples (default 25)
    sr: sampling rate, default 16000

    Returns:
    list of starting time (in samples unit)
    """
    approx_nb_samples = int((duration / duration_sample) ** 0.5) + 1
    samples = np.random.randint(low=0, high=(duration - duration_sample) * sr, size=(approx_nb_samples))
    samples = np.sort(samples)
    # overlap if less than min_separation apart
    overlapping = np.concatenate(([False], (samples[1:] - samples[:-1]) < min_separation * sr))
    return samples[~overlapping]  # remove overlapping samples

def get_database_name(accent, url):
    return DIR_DATABASE + accent + '/' + url.replace("https://www.youtube.com/watch?v=", "") + ".mp3"

def download_audio(accent: str, video_dict: dict) -> None:
    """ Download audio tracks from video_dict using yt-dlp.
    """
    to_download_urls = []

    # list files to download
    for category in video_dict[accent]:
        for url in video_dict[accent][category]:
            if not os.path.isfile(get_database_name(accent, url)):
                to_download_urls.append(url)

    print(f"{accent}: downloading {len(to_download_urls)} files")
    for url in tqdm(to_download_urls[:1]):
        output_name = get_database_name(accent, url)
        command = f'yt-dlp -o {output_name} -x --audio-format mp3 --postprocessor-args "-ac 1 -ar 16000" {url}'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE
                                   )
        stdout, stderr = process.communicate()

def process_audio(accent):
    """ Process audio files into 30s segments
    """

    dir_name = DIR_DATABASE + accent
    files_database = os.listdir(dir_name)
    print(f"{accent}: found {len(files_database)} files in {dir_name}")

    dir_name_out = DIR_DATASET + accent
    files_dataset = os.listdir(dir_name_out)
    files_dataset = [name[:11] for name in files_dataset]  # strip _i.mp3

    files_to_process = [file for file in files_database
                        if file[:11] not in files_dataset]

    if len(files_to_process) > 0:
        print(f"{len(files_to_process)} files to process")
    else:
        print("All files processed, nothing to do.")

    for file_name in files_to_process:
        path = dir_name + '/' + file_name
        duration = librosa.get_duration(path=path)
        offsets = select_samples_offsets(duration)
        for i, offset in enumerate(offsets):
            sr = 16e3
            audio, sr = librosa.load(path, sr=sr, offset=offset / sr, duration=30)
            path_out = dir_name_out + '/' + file_name.replace(".mp3", f"_{i}.mp3")
            sf.write(path_out, audio, int(sr), format='mp3')

if __name__ == "__main__":
    # Download files
    DIR_DATABASE = 'database/'
    DIR_DATASET = 'dataset/'

    with open(DIR_DATABASE + "videos_urls", "r") as f:
        video_dict = json.load(f)
    print("Found videos_urls. Checking for files to download.")

    accent_list = video_dict.keys()

    for accent in accent_list:
        download_audio(accent, video_dict)
        process_audio(accent)
