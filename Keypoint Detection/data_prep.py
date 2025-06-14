from dataset import ClassDataset
from torch.utils.data import DataLoader

def get_data_loaders(train_transform=None, val_transform=None,
                     train_path='',
                     val_path='',
                     test_path=''):
    dataset_train = ClassDataset(train_path, transform=train_transform() if train_transform else None, demo=False)
    dataset_val = ClassDataset(val_path, transform=val_transform() if val_transform else None, demo=False)
    dataset_test = ClassDataset(test_path, transform=val_transform() if val_transform else None, demo=False)
    data_loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True, collate_fn=dataset_train.collate_fn, num_workers=2, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, collate_fn=dataset_val.collate_fn, num_workers=2, pin_memory=True)
    data_loader_test = DataLoader(dataset_test, batch_size=4, shuffle=False, collate_fn=dataset_test.collate_fn, num_workers=2, pin_memory=True)
    return {
        'train': data_loader_train,
        'val': data_loader_val,
        'test': data_loader_test,
    }
