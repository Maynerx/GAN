import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.file_list = self.load_file_list()

    def load_file_list(self):
        return [f'{self.data_path}/{i}' for i in os.listdir(self.data_path)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = self.file_list[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image