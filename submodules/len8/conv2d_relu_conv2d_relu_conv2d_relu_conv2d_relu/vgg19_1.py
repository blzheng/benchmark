import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d8 = Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu8 = ReLU(inplace=True)
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu9 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu11 = ReLU(inplace=True)

    def forward(self, x19):
        x20=self.conv2d8(x19)
        x21=self.relu8(x20)
        x22=self.conv2d9(x21)
        x23=self.relu9(x22)
        x24=self.conv2d10(x23)
        x25=self.relu10(x24)
        x26=self.conv2d11(x25)
        x27=self.relu11(x26)
        return x27

m = M().eval()
x19 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
