import torch
import torch.nn as nn
import torch.nn.parallel



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# the leafnet is defined as
# https://www.kaggle.com/abhmul/keras-convnet-lb-0-0052-w-visualization
class leafnet(nn.Module):
    def __init__(self, nc):
        super(leafnet, self).__init__()
        self.nc = nc
        self.conv = nn.Sequential(
            nn.Conv2d(self.nc, 8, 5, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 32, 5, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(14304, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 99),
            nn.Softmax()
        )


    def forward(self, input_img, input_att, nb):
        img_conv = self.conv.forward(input_img)
        img_flat = img_conv.view(nb, -1)
        x = torch.cat((img_flat, input_att), 1)
        
        ##fully connect
        output = self.fc(x)
        return output
