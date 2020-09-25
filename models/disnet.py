import torch
import torch.nn as nn
import torch.nn.functional as F

class disnet_lr(nn.Module):
    def __init__(self,num_classes=1):
        super(disnet, self).__init__()
        #self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Sequential(
                      nn.Linear(4096, 1024),
                      nn.BatchNorm1d(1024),
                      nn.LeakyReLU(0.2, inplace = True)
                      )
        self.fc3 = nn.Sequential(
                      nn.Linear(1024, 512),
                      nn.BatchNorm1d(512),
                      nn.LeakyReLU(0.2, inplace = True)
                      )
        self.fc4 = nn.Linear(512, 1)
                   
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = x.view(x.size(0),-1)
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = nn.Sigmoid()(self.fc4(x))

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class disnet(nn.Module):
    def __init__(self,num_classes=1):
        super(disnet, self).__init__()
        #self.fc1 = nn.Linear(4096, 4096)
        self.fc2 = nn.Sequential(
                      nn.Linear(4096, 1024),
                      #nn.BatchNorm1d(1024),
                      #nn.ReLU(0.2, inplace = True)
                      )
        self.fc3 = nn.Sequential(
                      nn.Linear(1024, 512),
                      #nn.BatchNorm1d(512),
                      #nn.ReLU(0.2, inplace = True)
                      )
        self.fc4 = nn.Linear(512, 1)
                   
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #x = x.view(x.size(0),-1)
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class dis_conv_net(nn.Module):
    def __init__(self,num_classes=1):
        super(dis_conv_net, self).__init__()
        
        self.conv1 = nn.Sequential(
                     nn.Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),             
                     nn.LeakyReLU(inplace=True),
                     nn.Dropout2d(0.25), 
        )
        self.conv2 = nn.Sequential(
                     nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),             
                     nn.LeakyReLU(inplace=True), 
                     nn.Dropout2d(0.25)
        )
        self.flatten = Flatten()
        self.fc1 = nn.Linear(32*7*7, 512)
        self.fc2 = nn.Sequential(nn.Linear(512,1),nn.Sigmoid())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)        
        x = self.conv2(x)       
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
