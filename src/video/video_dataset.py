# -*- coding: utf-8 -*-
import os
import pandas as pd
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Required constants.
# ROOT_DIR = os.path.join("../..", "input", "Human Action Recognition", "train")
ROOT_DIR = os.path.join("C", "work", "repo", "Datasets", "magic_fall_dataset")
CSV_PATH = os.path.join("magic_fall_dataset_2class_train.csv")
TRAIN_RATIO = 85
VALID_RATIO = 100 - TRAIN_RATIO
FRAME_SIZE = 224  # Frame size of resize when applying transforms.
NUM_WORKERS = 12  # Number of parallel processes for data preparation.
SEQUENCE_LENGTH = 16  # Number of frames to sample from each video


# Training transforms
def get_train_transform(frame_size):
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((frame_size, frame_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(35),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            transforms.GaussianBlur(kernel_size=3),
            transforms.RandomGrayscale(p=0.5),
            transforms.RandomRotation(45),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform


# Validation transforms
def get_valid_transform(frame_size):
    valid_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return valid_transform


def shuffle_csv():
    df = pd.read_csv(CSV_PATH, sep=",")
    df = df.sample(frac=1)
    num_train = int(len(df) * (TRAIN_RATIO / 100))
    num_valid = int(len(df) * (VALID_RATIO / 100))

    train_df = df[:num_train].reset_index(drop=True)
    valid_df = df[-num_valid:].reset_index(drop=True)
    return train_df, valid_df


class VideoDataset(Dataset):
    def __init__(self, df, class_names, is_train=False):
        self.video_dir = ROOT_DIR
        self.df = df
        self.video_names = self.df.filename
        self.labels = list(self.df.label)
        self.class_names = class_names
        self.sequence_length = SEQUENCE_LENGTH
        if is_train:
            self.transform = get_train_transform(FRAME_SIZE)
        else:
            self.transform = get_valid_transform(FRAME_SIZE)

    def __len__(self):
        return len(self.video_names)

    def load_video(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Calculate frame indices to sample
        if self.sequence_length >= total_frames > 0:
            # If video is shorter than desired sequence, duplicate frames
            indices = [x % total_frames for x in range(self.sequence_length)]
        else:
            # Randomly sample frames if training, or take uniform samples if validation
            step = total_frames // self.sequence_length
            indices = [i * step for i in range(self.sequence_length)]

        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(frame))
            else:
                # If frame reading fails, create a blank frame
                frames.append(torch.zeros(3, FRAME_SIZE, FRAME_SIZE))

        cap.release()
        # Stack frames along a new dimension
        return torch.stack(frames)

    def __getitem__(self, index):
        video_path = os.path.join(self.video_dir, self.video_names[index])
        label = self.labels[index]

        # Load and process video
        video_tensor = self.load_video(video_path)
        class_num = self.class_names.index(label)

        return {
            "video": video_tensor,  # Shape: [sequence_length, channels, height, width]
            "label": class_num,
        }


def get_data_loaders(dataset_train, dataset_valid, batch_size):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    return train_loader, valid_loader
