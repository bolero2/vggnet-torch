import math
import time
import cv2
import os

import numpy as np
import torch
from torch.nn.functional import softmax
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from datasets import CustomDataset
import logging


logger = logging.getLogger(__name__)

class VGGNet(nn.Module):
    def __init__(self, name, ch, num_classes, setting=None):
        super(VGGNet, self).__init__()
        # =========================== Setting ============================
        self.yaml = setting
        self.img_size = self.yaml['img_size']

        self.num_classes = num_classes
        self.category_names = [str(x) for x in range(0, self.num_classes)]
        self.root_dir = self.yaml['DATASET']['root_path']
        self.ext = self.yaml['DATASET']['ext']

        self.conv_layers = list()
        self.flatten = list()
        self.fc_layers = list()

        # ======================== get layer info ========================
        self.name = name
        layerset = self.yaml[self.name]
        fcset = self.yaml['fc_layer']
        self.ch = ch

        # ======================= Model Definition =======================
        for block in layerset:
            for layer_output in block:
                self.conv_layers += [nn.Conv2d(self.ch, layer_output, kernel_size=3, padding=1), 
                                      nn.BatchNorm2d(layer_output), 
                                      nn.ReLU(inplace=True)]
                self.ch = layer_output
            self.conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        self.conv_layers += [nn.AdaptiveAvgPool2d((7, 7))]

        self.flatten += [nn.Flatten()]

        last_block = 0
        for block in range(0, len(fcset)):
            if block == 0:
                self.fc_layers += [nn.Linear(512 * 7 * 7, fcset[block])]
            else:
                self.fc_layers += [nn.Linear(fcset[block - 1], fcset[block])]
            last_block = fcset[block]

        self.fc_layers += [nn.Linear(last_block, self.num_classes)]

        self.total_layers = self.conv_layers + self.flatten + self.fc_layers

        self.model = nn.Sequential(*self.total_layers)
        self.softmax = softmax
        # ================================================================

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, phase='train'):
        for layer in self.model:
            x = layer(x)

        if phase == 'train':
            return x

        elif phase == 'test':
            return self.softmax(x)
            
        else:
            raise ValueError(f"phase: [{phase}] is invalid mode.")

    def fit(self, x, y, validation_data, epochs=30, batch_size=4):
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda else torch.device('cpu')
        imgsz = self.img_size

        trainpack = (x, y)
        validpack = validation_data[0:2]

        # Batch size 재조정
        min_batch_size = len(x)
        if batch_size > min_batch_size:
            logger.info("Batch size will be overwritten from {} to {}.".format(batch_size, min_batch_size))
            batch_size = min_batch_size

        # num_workers(cpu) 재조정
        if self.yaml['workers'] < batch_size:
            self.yaml['workers'] = batch_size if batch_size <= 4 else 4

        train_dataset = CustomDataset(trainpack, 
                                      datadir=self.root_dir, 
                                      category_names=self.category_names, 
                                      imgsz=imgsz)

        valid_dataset = CustomDataset(validpack, 
                                      datadir=self.root_dir, 
                                      category_names=self.category_names, 
                                      imgsz=imgsz)

        total_train_iter = math.ceil(len(x) / batch_size)
        total_valid_iter = math.ceil(len(validation_data[0]) / batch_size)

        trainloader = DataLoader(train_dataset,
                                 batch_size=batch_size, 
                                 num_workers=self.yaml['workers'], 
                                 shuffle=False,
                                 pin_memory=True)

        validloader = DataLoader(valid_dataset, 
                                #  batch_size=int(batch_size / 2), 
                                 batch_size=batch_size,
                                 num_workers=self.yaml['workers'], 
                                 shuffle=False,
                                 pin_memory=True)

        self.optimizer = optim.SGD(self.parameters(), lr=1e-4, momentum=0.9)

        scheduler = ReduceLROnPlateau(self.optimizer,
                                      verbose=True,
                                      factor=float(self.yaml['TRAIN']['factor']),
                                      patience=int(self.yaml['TRAIN']['factor']))

        self = self.to(_device)
        
        for epoch in range(0, epochs):
            epoch_start = time.time()
            train_loss, train_acc = 0, 0
            self.train()
            # Training Part
            print(f"[Epoch {epoch + 1}/{epochs}] Start")
            for i, (img, label) in enumerate(trainloader):
                iter_start = time.time()
                img = Variable(img.to(_device))
                label = Variable(label.to(_device, dtype=torch.int64))

                out = self(img, phase='train')
                acc = (torch.max(out, 1)[1].cpu().numpy() == torch.max(label, 1)[1].cpu().numpy())
                acc = float(np.count_nonzero(acc) / batch_size)
                loss = self.criterion(out, torch.max(label, 1)[1])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_acc += acc
                print("[train %5s/%5s] Epoch: %4s | Time: %6.2fs | loss: %10.4f | Acc: %10.4f" % (
                        i + 1, total_train_iter, epoch + 1, time.time() - iter_start, round(loss.item(), 4), float(acc)))

            train_loss = train_loss / total_train_iter
            train_acc = train_acc / total_train_iter
            print("[Epoch {} training Ended] > Time: {:.2}s/epoch | Loss: {:.4f} | Acc: {:g}\n".format(
                epoch + 1, time.time() - epoch_start, np.mean(train_loss), train_acc))

            val_loss, val_acc = self.evaluate(model=self,
                                              dataloader=validloader,
                                              valid_iter=total_valid_iter,
                                              batch_size=batch_size,
                                              criterion=self.criterion)

            scheduler.step(round(val_loss, 4))

    def evaluate(self, model, dataloader, valid_iter=1, batch_size=1, criterion=None):
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda else torch.device('cpu')

        model.eval()
        model.to(_device)

        start = time.time()
        total_loss, total_acc = 0, 0
        with torch.no_grad():
            for i, (img, label) in enumerate(dataloader):
                img = Variable(img.to(_device))
                label = Variable(label.to(_device, dtype=torch.int64))

                out = model(img, phase='train')

                loss = criterion(out, torch.max(label, 1)[1])
                acc = (torch.max(out, 1)[1].cpu().numpy() == torch.max(label, 1)[1].cpu().numpy())
                acc = float(np.count_nonzero(acc) / batch_size)

                total_loss += loss
                total_acc += acc

                print("[valid {}/{}] Time: {:.2}s | loss: {} | Acc: {}".format(
                    i + 1, valid_iter, time.time() - start, round(loss.item(), 4), float(acc)))
        total_loss = total_loss / valid_iter
        total_acc = total_acc / valid_iter

        return total_loss.item(), total_acc
        
    def predict(self, test_images, use_cpu=False):
        is_cuda = torch.cuda.is_available()
        _device = torch.device('cuda') if is_cuda and not use_cpu else torch.device('cpu')

        self.eval()
        self.to(_device)
        result_np = []
        
        for test_image in test_images:
            img = cv2.imread(test_image, cv2.IMREAD_COLOR)
            img = cv2.resize(img, dsize=(self.img_size[0], self.img_size[1]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32)
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            img = torch.Tensor(img).permute(0, 3, 1, 2).to(_device)
            # print(img.shape)
            pred = self(img, phase='test')
            pred_np = pred.cpu().detach().numpy()
            for elem in pred_np:
                result_np.append(elem)
        result_np = np.array(result_np)
        return result_np


def num(s):
    """ 3.0 -> 3, 3.001000 -> 3.001 otherwise return s """
    s = str(s)
    try:
        int(float(s))
        return s.rstrip('0').rstrip('.')
    except ValueError as e:
        return s