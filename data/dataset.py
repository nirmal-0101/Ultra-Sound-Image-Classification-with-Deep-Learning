#dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

class ImageCustomDataset(Dataset):
    """
    A custom dataset for loading the ultrasound images
    and labels from the dataframe.
    """
    def __init__(self, data, images_path, transform=None):
        self.data = data.reset_index(drop=True)
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data.loc[index, 'Image_name'] + '.png'
        label = self.data.loc[index, 'plane_class']
        image_path = os.path.join(self.images_path, image_name)

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
