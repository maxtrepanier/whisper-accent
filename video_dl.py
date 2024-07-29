#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Builds an audio dataset by fetching videos from given urls, extracting audio
clips of given duration, and performing a train-test split.
"""

__author__  = "Maxime Trepanier"

# I/O
import json
import os
import subprocess
from tqdm import tqdm

# Processing
import re
import random

# Sound processing
import numpy as np
import librosa
import soundfile as sf

# Python file
from typing import List, Dict

def get_database_name(dir_name: str, url: str) -> str:
    """ Returns file name by removing youtube url prefix and appending file
    extension.

    Args:
    - dir_name: name of the directory in which the file is located
    - url: url address to download
    Returns:
    - file name (including path)
    """
    if dir_name[-1] != '/':
        dir_name += '/'
    return dir_name + url.replace("https://www.youtube.com/watch?v=", "") + ".mp3"

def get_name_from_sample(sample_name):
    """ Returns sample_name without the postfix _{i}.mp3.
    """
    sample_postfix = re.search("_\d+\.mp3", sample_name)
    if sample_postfix is None:
        raise ValueError(f"Cannot process {sample_name}, wrong format.")
    name = sample_name[:sample_postfix.start()]
    return name

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
    return samples[~overlapping]  # drop overlapping samples

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

def process_audio(accent: str, min_separation:float = 30, alpha: float = 0.5):
    """ Process audio files for given accent.
    For each file in DIR_DATABASE/accent, selects random segments of
    min_separation (default 30s) from the audio track, and outputs the segments
    to DIR_DATASET/DIR_SETS[0]/accent.

    Args:
    - accent: name of directory in DIR_DATABASE containing audio files
    - min_separation: minimum separation (in seconds) between the beginning of
      different samples
    - alpha: controls the density of samples, see select_samples_offsets for
      more details.
    Returns: None
    """

    dir_name = DIR_DATABASE + accent

    files_database = [f[:-4] for f in os.listdir(dir_name) if f[-4:] == ".mp3"]
    print(f"{accent}: found {len(files_database)} mp3 files in {dir_name}")

    dir_name_out = DIR_DATASET + DIR_SETS[0] + '/' + accent
    if not os.path.exists(dir_name_out):
        os.makedirs(dir_name_out)

    files_dataset = [f for f in os.listdir(dir_name_out) if f[-4:] == ".mp3"]
    for dir_name in DIR_SETS[1:]:
        dir_name_full = DIR_DATASET + dir_name + '/' + accent
        if not os.path.exists(dir_name_full):
            os.makedirs(dir_name_full)
        files_dataset += [f for f in os.listdir(dir_name_full)
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

class File:
    """ Container class for an audio track and its samples.
    """
    def __init__(self, dir_name: str, name: str) -> None:
        """ Create a new container for a file with [name] in [dir_name].
        Args:
         - dir_name: name of the directory where the file is located
         - name: name of the file
        """
        self.name = name
        self.dir_name = dir_name
        self.samples = []  # holds name of samples

    def add_sample(self, sample_name: str):
        """ Links a sample with name [sample_name] (assumed in [dir_name]) to
        the file.

        Args:
         - sample_name: name of the sample
        """
        self.samples.append(sample_name)

    def nb_samples(self): return len(self.samples)

    def get_samples_path(self) -> [str, str]:
        """ Returns a list of samples and their path.

        Returns:
         - list [(sample_name, sample_path)] of all samples
        """
        samples_path = [(f, self.dir_name + '/' + f) for f in self.samples]
        return samples_path

    def __str__(self):
        return self.name + f" ({self.nb_samples()} samples)"

    def __repr__(self):
        return self.__str__()


def get_dataset_dict(dir_name: str, files_dataset: List[str]) -> Dict[str, File]:
    """ Builds a dataset dictionary from a list of files located in directory
    [dir_name].

    Args:
     - dir_name: name of the directory
     - files_dataset: list of files within dir_name to process
    Returns:
     - dict: keys are file names, values are File objects containing samples.
    """

    dataset = {}
    for file in files_dataset:
        file_name = get_name_from_sample(file)
        if file_name not in dataset:
            dataset[file_name] = File(dir_name, file_name)

        dataset[file_name].add_sample(file)

    return dataset

def train_test_split(accent, test_ratio: float = 0.15, tol: float = 0.02):
    """ Divides audio samples into train/test with given ratio of test samples.
    """

    # retrieve file list in directories
    dir_name_assigned = [DIR_DATASET + d + '/' + accent for d in DIR_SETS]

    for d in dir_name_assigned:
        if not os.path.exists(d):
            os.makedirs(d)

    files_dataset_assigned = {d: [f for f in os.listdir(d) if f[-4:] == ".mp3"]
                              for d in dir_name_assigned}


    nb_files_assigned = {d: len(files_dataset_assigned[d]) for d in files_dataset_assigned}
    nb_total_files = sum(nb_files_assigned.values())
    print(f"{accent}: found {nb_total_files} mp3 files in {DIR_DATASET}./{accent}/:")
    for d in dir_name_assigned:
        print(f"  - {nb_files_assigned[d]} in {d}")
    print("")

    # check ratios of test samples
    resplit = False
    if nb_total_files > 0:
        # if assigned already, check if within tol of target
        if abs(nb_files_assigned[dir_name_assigned[1]] / nb_total_files -
                test_ratio) > tol:
            resplit = True

    if not resplit:
        print(f"Train-test split already done, within {tol*100}% of target {test_ratio}. Nothing to do.\n")
        return

    dataset = {}
    # merge assigned file to dataset
    for d in dir_name_assigned:
        dataset = dataset | get_dataset_dict(d, files_dataset_assigned[d])

    # assign test files using greedy approach
    target_test = test_ratio * nb_total_files
    error_margin = tol * nb_total_files
    shuffle_files = list(dataset.keys())
    random.shuffle(shuffle_files)  # randomize files
    test_files = []
    nb_test_samples = 0

    for file in shuffle_files:
        if nb_test_samples >= target_test - error_margin:  # reached target
            break

        nb_samples = dataset[file].nb_samples()
        if nb_test_samples + nb_samples <= target_test + error_margin:
            test_files.append(file)
            nb_test_samples += nb_samples

    # assign rest of files to train set
    train_files = [f for f in shuffle_files if f not in test_files]

    # move file to appropriate folder
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

    DIR_SETS = ['train', 'dev']  # names of sets. first is training, second is dev set

    with open(DIR_DATABASE + "videos_urls", "r") as f:
        video_dict = json.load(f)
    print("Found videos_urls. Checking for files to download.")

    accent_list = video_dict.keys()

    for accent in accent_list:
        download_audio(video_dict, accent)
        process_audio(accent, alpha=0.8)
        train_test_split(accent, test_ratio=0.2)
