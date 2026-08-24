from torch.utils.data import DataLoader
from torchvision import transforms
from MAPS.integration_P.datasets.data_loader import *


def liver_dataloader(args, dataset):

    trainloader = DataLoader(
                liver_loader(dataset.datasets, transform=None),
                batch_size=args.train_batch, 
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=True,
                persistent_workers=True
            )

    testloader = DataLoader(
                liver_loader(dataset.datasets_test, transform=None),
                batch_size=args.test_batch, 
                shuffle=False, 
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=False,
                persistent_workers=True
                )
    return trainloader, testloader

def liver_dataloader_inference(args, dataset):

    testloader = DataLoader(
                liver_loader(dataset.datasets + dataset.datasets_test, transform=None),
                batch_size=args.test_batch, 
                shuffle=False, 
                num_workers=args.workers,
                pin_memory=True, 
                drop_last=False,
                persistent_workers=True
                )
    return testloader
