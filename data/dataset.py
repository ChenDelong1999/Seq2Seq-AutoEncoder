
from torchvision.datasets import CIFAR10, CIFAR100, STL10, MNIST
from torchvision.transforms import ToTensor
from .image_classification_dataset import SeqImgClsDataset
from .segment_dataset import COCOMaskDataset

def get_dataset(args):
    if args.dataset=='cifar10':
        train_dataset = SeqImgClsDataset(
            dataset=CIFAR10(root=args.data_dir, train=True, download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=3,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
        test_dataset = SeqImgClsDataset(
            dataset=CIFAR10(root=args.data_dir, train=False, download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=3,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
    elif args.dataset=='cifar100':
        train_dataset = SeqImgClsDataset(
            dataset=CIFAR100(root=args.data_dir, train=True, download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=3,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
        test_dataset = SeqImgClsDataset(
            dataset=CIFAR100(root=args.data_dir, train=False, download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=3,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
    elif args.dataset=='stl10':
        train_dataset = SeqImgClsDataset(
            dataset=STL10(root=args.data_dir, split='train+unlabeled', download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=3,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
        test_dataset = SeqImgClsDataset(
            dataset=STL10(root=args.data_dir, split='test', download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=3,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
    elif args.dataset=='mnist':
        train_dataset = SeqImgClsDataset(
            dataset=MNIST(root=args.data_dir, train=True, download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=1,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
        test_dataset = SeqImgClsDataset(
            dataset=MNIST(root=args.data_dir, train=False, download=True, transform=ToTensor()),
            img_size=args.img_size,
            img_channels=1,
            num_queries=args.num_queries,
            min_resize_ratio=args.min_resize_ratio,
        )
    elif args.dataset=='coco':
        train_dataset = COCOMaskDataset(coco_root=args.data_dir, split='train', num_queries=args.num_queries, virtual_dataset_size=100000, data_seq_length=args.img_size**2, min_pixel_num=16)
        test_dataset = COCOMaskDataset(coco_root=args.data_dir, split='val', num_queries=args.num_queries, virtual_dataset_size=100000, data_seq_length=args.img_size**2, min_pixel_num=16)

    return train_dataset, test_dataset

