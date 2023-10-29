import os
import argparse
import logging
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


class DogBreedClassification:
    IMG_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, args):
        self.data_dir = args.data
        self.train_data_path = os.path.join(self.data_dir, 'train')
        self.validation_data_path = os.path.join(self.data_dir, 'valid')
        self.test_data_path = os.path.join(self.data_dir, 'test')

        self.batch_size = args.batch_size
        self.lr = args.lr

    def create_model(self, device):
        '''Use pretrained MobileNetV2 model'''

        model = models.mobilenet_v2(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        model.classifier[1] = nn.Sequential(
            nn.Linear(in_features=1280, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=133)
        )
        model = model.to(device)

        return model

    def create_data_loader(self):
        train_transform = transforms.Compose([
            transforms.Resize(size=self.IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.MEAN,
                std=self.STD
            )
        ])

        valid_transform = transforms.Compose([
            transforms.Resize(size=self.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.MEAN,
                std=self.STD
            )
        ])

        train_datasets = torchvision.datasets.ImageFolder(
            root=self.train_data_path,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_datasets,
            batch_size=self.batch_size,
            shuffle=True
        )

        validation_datasets = torchvision.datasets.ImageFolder(
            root=self.validation_data_path,
            transform=valid_transform,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_datasets,
            batch_size=self.batch_size,
            shuffle=False
        )

        test_datasets = torchvision.datasets.ImageFolder(
            root=self.test_data_path,
            transform=valid_transform
        )
        test_loader = torch.utils.data.DataLoader(
            test_datasets,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, validation_loader, test_loader

    def prepare_training(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.create_model(device)

        loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        return model, loss_criterion, optimizer, device

    def train(self,
              model,
              train_loader,
              validation_loader,
              criterion,
              optimizer,
              device,
              epochs=25):
        best_loss = 1e6
        image_dataloader = {'train': train_loader, 'valid': validation_loader}
        loss_counter = 0

        for epoch in tqdm(range(epochs)):
            logger.info(f"Epoch: {epoch}")
            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in image_dataloader[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / len(image_dataloader[phase].dataset)
                epoch_acc = running_corrects / len(image_dataloader[phase].dataset)

                if phase == 'valid':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                    else:
                        loss_counter += 1

                logger.info('Epoch {}/Phase {}, Loss: {:.4f}, Accuracy: {:.4f}, Best loss: {:.4f}'.format(
                    epoch,
                    phase,
                    epoch_loss,
                    epoch_acc,
                    best_loss)
                )

            if loss_counter == 1:
                break
            if epoch == 0:
                break

        return model

    def test(self,
             model,
             test_loader,
             criterion,
             device):
        model.eval()
        running_loss = 0
        running_corrects = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        total_loss = running_loss / len(test_loader.dataset)
        total_acc = running_corrects.double() / len(test_loader.dataset)

        logger.info(f"Testing Loss: {total_loss}")
        logger.info(f"Testing Accuracy: {total_acc}")


def main(args):

    # Setup
    classifier = DogBreedClassification(args)
    train_loader, validation_loader, test_loader = classifier.create_data_loader()
    model, loss_criterion, optimizer, device = classifier.prepare_training()

    # Train
    model = classifier.train(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criterion=loss_criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs
    )
    # Test
    classifier.test(model=model,
                    test_loader=test_loader,
                    criterion=loss_criterion,
                    device=device)

    # Save model
    model_path = os.path.join(args.model_output_dir, "model.pth")
    torch.save(model.cpu().state_dict(), model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dog Breed Classification")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        default=os.environ["SM_MODEL_DIR"]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.environ["SM_CHANNEL_TRAINING"]
    )
    args = parser.parse_args()

    main(args)