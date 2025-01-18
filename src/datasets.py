import os
import pandas as pd
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Required constants.
ROOT_DIR = os.path.join('..', 'input', 'Human Action Recognition', 'train')
CSV_PATH = os.path.join(
    '..', 'input', 'Human Action Recognition', 'Training_set.csv'
)
TRAIN_RATIO = 85
VALID_RATIO = 100 - TRAIN_RATIO
IMAGE_SIZE = 224  # Image size of resize when applying transforms.
NUM_WORKERS = 4  # Number of parallel processes for data preparation.


# Training transforms
def get_train_transform(image_size):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(35),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomRotation(45),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return train_transform


# Validation transforms
def get_valid_transform(image_size):
    valid_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return valid_transform


def shuffle_csv():
    df = pd.read_csv(CSV_PATH)
    df = df.sample(frac=1)
    num_train = int(len(df) * (TRAIN_RATIO / 100))
    num_valid = int(len(df) * (VALID_RATIO / 100))

    train_df = df[:num_train].reset_index(drop=True)
    valid_df = df[-num_valid:].reset_index(drop=True)
    return train_df, valid_df


class CustomDataset(Dataset):
    def __init__(self, df, class_names, is_train=False):
        self.image_dir = ROOT_DIR
        self.df = df
        self.image_names = self.df.filename
        self.labels = list(self.df.label)
        self.class_names = class_names
        if is_train:
            self.transform = get_train_transform(IMAGE_SIZE)
        else:
            self.transform = get_valid_transform(IMAGE_SIZE)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_names[index])
        label = self.labels[index]
        # Process and transform images.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(image)
        class_num = self.class_names.index(label)
        return {
            'image': image_tensor,
            'label': class_num
        }
