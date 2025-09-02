#data_loader.py
from torch.utils.data import DataLoader
from data.dataset import ImageCustomDataset

def get_dataloaders(train_data, valid_data, test_data, images_path, transform_train, transform_valid, batch_size=64, num_workers=2):
    """
    Returns train, validation and test DataLoaders using custom ImageCustomDataset.
    """
    dataset_train = ImageCustomDataset(train_data, images_path, transform=transform_train)
    dataset_valid = ImageCustomDataset(valid_data, images_path, transform=transform_valid)
    dataset_test = ImageCustomDataset(test_data, images_path, transform=transform_valid)

    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return loader_train, loader_valid, loader_test
