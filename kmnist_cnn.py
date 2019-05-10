import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import torch.optim as optim

torch.set_default_tensor_type('torch.DoubleTensor')

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(in_features=1568, out_features=600)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=600, out_features=10)


    def forward(self,x):
        out = self.relu(self.bn1(self.cnn1(x)))
        out = self.maxpool(out)
        out = self.relu(self.bn2(self.cnn2(out)))
        out = self.maxpool(out)
        out = out.view(-1,1568)
        out = self.fc2(self.dropout(self.relu(self.fc1(out))))
        return out



def train(model, lr, beta1, train_loader, val_loader, epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = optim.Adam(model.parameters(),lr,betas=[beta1,0.999])
    criterion = nn.CrossEntropyLoss()
    val_accuracy = 0
    for e in range(epochs):
        for step, (train_images, train_labels) in enumerate(train_loader):
            print("Entering epoch " , e+1, " batch ", step, "with learning rate ", lr , "and beta1 ", beta1)
            opt.zero_grad()
            train_images = Variable(train_images)
            train_labels = Variable(train_labels)
            train_images = train_images.type(torch.DoubleTensor).to(device)
            train_labels = train_labels.type(torch.LongTensor).to(device)
            predictions = model.forward(train_images)
            loss = criterion(predictions,train_labels)
            loss.backward()
            print("Model Loss : ", float(loss))
            opt.step()

        torch.save(model, "Cnn_Kmnist.pt")

        with(torch.set_grad_enabled(False)):
            total = 0
            correct = 0
            for step, (val_images,val_labels) in enumerate(val_loader):
                total+= val_images.shape[0]
                val_images = Variable(val_images).type(torch.DoubleTensor).to(device)
                val_labels = Variable(val_labels).type(torch.LongTensor).to(device)
                _,val_predictions = torch.max(model.forward(val_images),dim=1)
                # if step == 1:
                #     print("val predictions " , val_predictions)
                #     print("val_labels" , val_labels)
                correct += torch.sum(val_predictions==val_labels)

            print("Total ", total)
            print("correct ", correct)

            val_accuracy = float(float(correct)/float(total))
            print("Validation Accuracy for epoch ", e+1, " : ", val_accuracy)

    return val_accuracy

    # def train_CNN(learning_rate, momentum, train_loader):
