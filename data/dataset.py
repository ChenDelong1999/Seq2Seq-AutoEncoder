
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from torchvision.transforms import ToTensor
from .image_classification_dataset import SeqImgClsDataset

def get_dataset(args):
    if args.dataset=='cifar10':
        train_dataset = SeqImgClsDataset(
            dataset=CIFAR10(root=args.data_dir, train=True, download=True, transform=ToTensor()),
            img_size=args.img_size,
            num_queries=args.num_queries,
        )
        test_dataset = SeqImgClsDataset(
            dataset=CIFAR10(root=args.data_dir, train=False, download=True, transform=ToTensor()),
            img_size=args.img_size,
            num_queries=args.num_queries,
        )
    elif args.dataset=='cifar100':
        train_dataset = SeqImgClsDataset(
            dataset=CIFAR100(root=args.data_dir, train=True, download=True, transform=ToTensor()),
            img_size=args.img_size,
            num_queries=args.num_queries,
        )
        test_dataset = SeqImgClsDataset(
            dataset=CIFAR100(root=args.data_dir, train=False, download=True, transform=ToTensor()),
            img_size=args.img_size,
            num_queries=args.num_queries,
        )
    elif args.dataset=='stl10':
        train_dataset = SeqImgClsDataset(
            dataset=STL10(root=args.data_dir, split='train+unlabeled', download=True, transform=ToTensor()),
            img_size=args.img_size,
            num_queries=args.num_queries,
        )
        test_dataset = SeqImgClsDataset(
            dataset=STL10(root=args.data_dir, split='test', download=True, transform=ToTensor()),
            img_size=args.img_size,
            num_queries=args.num_queries,
        )
    return train_dataset, test_dataset

