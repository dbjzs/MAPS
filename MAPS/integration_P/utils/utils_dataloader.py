from torch.utils.data import DataLoader
from torchvision import transforms
from MAPS.integration_P.datasets.data_loader import *


def dataloader_single(args, dataset):

    trainloader = DataLoader(
                single_omics_loader(dataset.datasets_train, transform=None),
                batch_size=args.train_batch, 
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=True,
                persistent_workers=True
            )

    testloader = DataLoader(
                single_omics_loader(dataset.datasets_test, transform=None),
                batch_size=args.test_batch, 
                shuffle=False, 
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=False,
                persistent_workers=True
                )
    return trainloader, testloader


def dataloader_multi(args, dataset):

    trainloader = DataLoader(
                multi_omics_loader(dataset.datasets_train, transform=None),
                batch_size=args.train_batch, 
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=True,
                persistent_workers=True
            )

    testloader = DataLoader(
                multi_omics_loader(dataset.datasets_test, transform=None),
                batch_size=args.test_batch, 
                shuffle=False, 
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=False,
                persistent_workers=True
                )
    return trainloader, testloader