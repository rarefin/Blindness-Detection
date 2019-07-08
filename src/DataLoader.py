from PIL import Image
from torch.utils.data import Dataset
from os.path import join


class TrainDataset(Dataset):
    def __init__(self, train_dir, train_df, transform):
        self.train_dir = train_dir
        self.train_df = train_df
        self.transform = transform

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        path = join(self.train_dir, self.train_df.loc[index, 'id_code'] + '.png')
        img = Image.open(path)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = int(self.train_df.loc[index, 'diagnosis'])

        return img, label


class TestDataset(Dataset):
    def __init__(self, test_dir, test_df, transform):
        self.test_dir = test_dir
        self.test_df = test_df
        self.transform = transform

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, index):
        path = join(self.test_dir, self.test_df.loc[index, 'id_code'] + '.png')
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return image