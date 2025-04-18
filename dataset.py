import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FruitDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir

        assert split in ["train", "test"], "split must be one of ['train',  'test']"
        self.split = split
        self.data_dir = os.path.join(root_dir, split)

        self.classes = os.listdir(self.data_dir)
        self.classes = sorted(self.classes)
        self.class_to_idx = {cls.lower(): idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls.lower() for idx, cls in enumerate(self.classes)}

        self.images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.data_dir, cls)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                self.images.append((img_path, self.class_to_idx[cls.lower()]))

        self.transform = transform
        if self.transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                ]
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
