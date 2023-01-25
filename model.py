import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from efficientnet_pytorch import EfficientNet
from tqdm.auto import tqdm


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print("Warning: GPU not available!")

data_path = Path("data/")
image_path = data_path / "20221206_classification_images/2022-12-06_classification_images_filtered_NIO"

data_transform = transforms.Compose([
    transforms.ToTensor()
])

data_set = datasets.ImageFolder(root=)


def layer1_multistream(num_cams, filter_num):
    """

    :param num_cams:
    :param filter_num:
    :return:
    """
    if not hasattr(layer1_multistream, "instance"):
        layer1_multistream.instance = 0
    j = layer1_multistream.instance
    seq = nn.Sequential(nn.Conv2d(in_channels=num_cams,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.BatchNorm2d(num_features=filter_num),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.BatchNorm2d(num_features=filter_num),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.ReLU(),
                        nn.Conv2d(in_channels=filter_num,
                                  out_channels=filter_num,
                                  kernel_size=(3, 3),
                                  ),
                        nn.BatchNorm2d(num_features=filter_num),
                        nn.ReLU()
                        )
    layer1_multistream.instance += 1
    return seq

def layer2_efficientnet(filter_num: int) -> nn.Sequential:
    """

    :param filter_num:
    :return:
    """
    seq = EfficientNet.from_name(model_name='efficientnet-b0',
                                 num_classes=2,
                                 in_channels=140)
    return seq

class EfficientnetLF(nn.Module):
    def __init__(self, view_num, filter_num):
        super().__init__()
        self.input_stack_vert = layer1_multistream(view_num, filter_num)
        self.input_stack_hori = layer1_multistream(view_num, filter_num)


    def forward(self, x):
        x = 1

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
        print("Warning: GPU not available!")
    network_name = 'EPIDef_train'
    iter00 = 0
    """
        Model parameters:
            first layer:  3 convolutional blocks for vertical and horizontal views
            second layer: modified EfficientNet
    """
    FILTER_NUMBER = 70
    dataset =
    BATCH_SIZE = 8
    pretrained_weights =
    LOAD_WEIGHTS = False
    EPOCHS = 1

    directory_ckp = f"efficientnet_lf_checkpoints\\{network_name}_ckp"
    if not os.path.exists(directory_ckp):
        os.makedirs(directory_ckp)
    if not os.path.exists('epidef_output\\'):
        os.makedirs('epidef_output\\')
    directory_t = f"epidef_output\\{network_name}"
    if not os.path.exists(directory_t):
        os.makedirs(directory_t)

    def train_step(model: torch.nn.Module,
                   dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer):
        model.train()
        train_loss, train_acc = 0, 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)
        return train_loss, train_acc

    def test_step(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module):
        model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                test_pred_logits = model(x)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)
        return test_loss, test_acc

    def train(model: torch.nn.Module,
              train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
              epochs = EPOCHS):
        results = {"train_loss": [],
                   "train_acc": [],
                   "test_loss": [],
                   "test_acc": []
                   }

        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                               dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer)
            test_loss, test_acc = test_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn)
            print(
                f"Epoch: {epoch + 1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
        return results


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
    print("Warning: GPU not available!")
network_name = 'EPIDef_train'
iter00 = 0
"""
    Model parameters:
        first layer:  3 convolutional blocks for vertical and horizontal views
        second layer: modified EfficientNet
"""
FILTER_NUMBER = 70
dataset =
BATCH_SIZE = 8
pretrained_weights =
LOAD_WEIGHTS = False
EPOCHS = 1

directory_ckp = f"efficientnet_lf_checkpoints\\{network_name}_ckp"
if not os.path.exists(directory_ckp):
    os.makedirs(directory_ckp)
if not os.path.exists('epidef_output\\'):
    os.makedirs('epidef_output\\')
directory_t = f"epidef_output\\{network_name}"
if not os.path.exists(directory_t):
    os.makedirs(directory_t)

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            test_pred_logits = model(x)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs = EPOCHS):
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results
