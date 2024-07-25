import json
import os
import subprocess
from tqdm import tqdm

import re
import random

import numpy as np
import librosa
import soundfile as sf

from typing import List, Dict

def select_samples_offsets(duration: int,
                           duration_sample: int = 30,
                           min_separation: int = 25,
                           alpha : float = 0.5,
                           sr: int = 16e3) -> List[int]:
    """ Select random samples (of duration_sample) from an audio track of given
    duration. The samples are separated by at least min_separation seconds.
    The parameter alpha controls how the number of samples drawn from a track
    scales with the duration of the track (recommended 0.5 < alpha < 1)

    Args:
    duration: total duration of the track (in seconds)
    duration_sample: duration of a single sample (default 30s)
    min_separation: minimal separation between samples (default 25)
    alpha: controls the number of samples
    sr: sampling rate, default 16000

    Returns:
    list of starting time (in samples unit)
    """
    approx_nb_samples = int((duration / duration_sample) ** alpha) + 1
    samples = np.random.randint(low=0, high=(duration - duration_sample) * sr, size=(approx_nb_samples))
    samples = np.sort(samples)
    # overlap if less than min_separation apart
    overlapping = np.concatenate(([False], (samples[1:] - samples[:-1]) < min_separation * sr))
    return samples[~overlapping]  # remove overlapping samples

def get_database_name(path, url):
    return path + url.replace("https://www.youtube.com/watch?v=", "") + ".mp3"

def download_audio(video_dict: dict, accent: str) -> None:
    """ Download audio tracks from video_dict using yt-dlp.

    video_dict is a dictionary with the structure
    {accent: {category: [url1, url2, ...]}}
    The function processes only the given accent.

    The audio tracks are extracted from the given url, converted to mono channel
    and resampled at 16kHz. They are stored in DIR_DATABASE/accent.

    Args:
    video_dict: dictionary containing urls to download
    accent: key of video_dict to process

    Returns: None
    """
    to_download_urls = []

    dir_name = DIR_DATABASE + accent + '/'
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # list files to download
    for category in video_dict[accent]:
        for url in video_dict[accent][category]:
            if not os.path.isfile(get_database_name(dir_name, url)):
                to_download_urls.append(url)

    print(f"{accent}: downloading {len(to_download_urls)} files")
    for url in tqdm(to_download_urls):
        output_name = get_database_name(dir_name, url)
        command = f'yt-dlp -o {output_name} -x --audio-format mp3 --postprocessor-args "-ac 1 -ar 16000" {url}'
        process = subprocess.Popen(command, shell=True,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE
                                   )
        stdout, stderr = process.communicate()

    print("Done.\n")

def process_audio(accent: str, min_separation=30, alpha=0.5):
    """ Process audio files for given accent.
    For each file in DIR_DATABASE/accent, selects random segments of 30s from
    the audio track, and outputs the segments to DIR_DATASET/accent.

    Args:
    accent: name of directory in DIR_DATABASE containing audio files
    Returns: None
    """

    dir_name = DIR_DATABASE + accent

    files_database = [f[:-4] for f in os.listdir(dir_name) if f[-4:] == ".mp3"]
    print(f"{accent}: found {len(files_database)} mp3 files in {dir_name}")

    dir_name_out = DIR_DATASET + accent
    if not os.path.exists(dir_name_out):
        os.makedirs(dir_name_out)

    files_dataset = [f for f in os.listdir(dir_name_out) if f[-4:] == ".mp3"]
    for dir_name in DIR_SETS:
        files_dataset += [f for f in os.listdir(dir_name_out + '/' + dir_name)
                if f[-4:] == ".mp3"]

    files_dataset = [get_name_from_sample(name) for name in files_dataset]  # strip _i.mp3

    files_to_process = [file for file in files_database
                        if file not in files_dataset]

    if len(files_to_process) > 0:
        print(f"{len(files_to_process)} files to process")
    else:
        print("All files processed, nothing to do.")

    for file_name in tqdm(files_to_process):
        path = dir_name + '/' + file_name
        duration = librosa.get_duration(path=path)
        offsets = select_samples_offsets(duration, min_separation=min_separation, alpha=alpha)
        for i, offset in enumerate(offsets):
            sr = 16e3
            audio, sr = librosa.load(path, sr=sr, offset=offset / sr, duration=30)
            path_out = dir_name_out + '/' + file_name.replace(".mp3", f"_{i}.mp3")
            sf.write(path_out, audio, int(sr), format='mp3')

    print("Done.\n")

def get_name_from_sample(sample_name):
    sample_postfix = re.search("_\d+\.mp3", sample_name)
    if sample_postfix is None:
        raise ValueError(f"Cannot process {sample_name}")
    name = sample_name[:sample_postfix.start()]
    return name

class File:
    def __init__(self, dir_name, name):
        self.name = name
        self.dir_name = dir_name
        self.samples = []

    def add_sample(self, sample_name):
        self.samples.append(sample_name)

    def nb_samples(self): return len(self.samples)

    def get_samples_path(self):
        samples_path = [(f, self.dir_name + '/' + f) for f in self.samples]
        return samples_path

    def __str__(self):
        return self.name + f" ({self.nb_samples()} samples)"

    def __repr__(self):
        return self.__str__()


def get_dataset_dict(dir_name: str, files_dataset: List[str]) -> Dict[str, File]:
    dataset = {}
    for file in files_dataset:
        file_name = get_name_from_sample(file)
        if file_name not in dataset:
            dataset[file_name] = File(dir_name, file_name)

        dataset[file_name].add_sample(file)

    return dataset

def train_test_split(accent, test=0.15):
    """ Divides audio samples into train/test with given ratio.
    """

    dir_name_base = DIR_DATASET + accent
    dir_name_assigned = [dir_name_base + '/' + d for d in DIR_SETS]

    for d in dir_name_assigned:
        if not os.path.exists(d):
            os.makedirs(d)

    files_dataset_base = [f for f in os.listdir(dir_name_base) if f[-4:] == ".mp3"]
    files_dataset_assigned = {d: [f for f in os.listdir(d) if f[-4:] == ".mp3"]
                              for d in dir_name_assigned}


    nb_files_assigned = {d: len(files_dataset_assigned[d]) for d in files_dataset_assigned}
    nb_files_unassigned = len(files_dataset_base)
    nb_total_files = nb_files_unassigned  + sum(nb_files_assigned.values())
    print(f"{accent}: found {nb_total_files} mp3 files in {dir_name_base}/:")
    print(f"  - {nb_files_unassigned} in {dir_name_base}")
    for d in dir_name_assigned:
        print(f"  - {nb_files_assigned[d]} in {d}")
    print("")

    resplit = False
    if nb_files_unassigned > 0:
        resplit = True
    elif nb_total_files > 0:
        # if assigned already, check if within 2% of target
        if abs(nb_files_assigned[dir_name_assigned[1]] / nb_total_files - test) > 0.02:
            resplit = True

    if not resplit:
        print(f"Train-test split already done, within 2% of target {test}. Nothing to do.\n")

    dataset = get_dataset_dict(dir_name_base, files_dataset_base)
    # merge assigned file to dataset
    for d in dir_name_assigned:
        dataset = dataset | get_dataset_dict(d, files_dataset_assigned[d])

    target_test = test * nb_total_files
    error_margin = 0.02 * nb_total_files
    shuffle_files = list(dataset.keys())
    random.shuffle(shuffle_files)
    test_files = []
    nb_test_samples = 0

    for file in shuffle_files:
        if nb_test_samples >= target_test - error_margin:
            break

        nb_samples = dataset[file].nb_samples()
        if nb_test_samples + nb_samples <= target_test + error_margin:
            test_files.append(file)
            nb_test_samples += nb_samples

    train_files = [f for f in shuffle_files if f not in test_files]

    for file in test_files:
        for sample, sample_path in dataset[file].get_samples_path():
            new_path = dir_name_assigned[1] + '/' + sample
            os.replace(sample_path, new_path)

    for file in train_files:
        for sample, sample_path in dataset[file].get_samples_path():
            new_path = dir_name_assigned[0] + '/' + sample
            os.replace(sample_path, new_path)

    print(f"Assigned {nb_test_samples} samples to dev set ({nb_test_samples / nb_total_files * 100:.2f}%)")
    return


if __name__ == "__main__":
    DIR_DATABASE = 'database/'
    DIR_DATASET = 'dataset/'

    DIR_SETS = ['train', 'dev']

    with open(DIR_DATABASE + "videos_urls", "r") as f:
        video_dict = json.load(f)
    print("Found videos_urls. Checking for files to download.")

    accent_list = video_dict.keys()

    for accent in accent_list:
        download_audio(video_dict, accent)
        process_audio(accent, alpha=0.8)
        train_test_split(accent, test=0.15)
