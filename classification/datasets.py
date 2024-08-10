import os
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from numpy.random import choice
from tqdm import tqdm

transform = transforms.Compose([transforms.ToTensor()])
class CIFAR10(Dataset):
    def __init__(
            self, root='/Users/sasha/ai_projects/data/CIFAR-10', train=True,
            num_labeled_per_class = 10
    ):
        # TODO: add random seed for predictable randomization
        # TODO: add opportunity to get only small labeled dataset
        dir = os.path.join(root, 'train' if train else 'test')
        self.classes = [cls for cls in os.listdir(dir) if '.' not in cls]
        self.num_labeled_per_class = num_labeled_per_class

        # create list of images
        self.imgs = []
        self.labels = dict()
        self.idx_to_class = dict()
        for i, class_ in enumerate(self.classes):
            imgs = [os.path.join(dir, class_, img) for img in os.listdir(os.path.join(dir, class_))]
            self.imgs.extend(imgs)
            # add labels
            self.idx_to_class[i] = class_
            imgs_iter = choice(imgs, self.num_labeled_per_class, False) if train else imgs
            for img_name in imgs_iter:
                self.labels[img_name] = i
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        # load images and targets
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        target = self.labels.get(img_path, -1)
        return transform(img), target

def test():
    data = CIFAR10()
    data_loader = DataLoader(
        data, batch_size = 4, shuffle = True, num_workers = 6, drop_last=True, pin_memory = True
    )
    loop = tqdm(data_loader, leave = True)
    for batch_idx, (imgs, labels) in enumerate(loop):
        loop.set_postfix(imgs_shape=imgs.shape, lables_shape = labels.shape)

if __name__ == '__main__':
    test()
