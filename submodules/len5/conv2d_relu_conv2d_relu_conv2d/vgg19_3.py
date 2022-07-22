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
        self.conv2d9 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu9 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x21):
        x22=self.conv2d9(x21)
        x23=self.relu9(x22)
        x24=self.conv2d10(x23)
        x25=self.relu10(x24)
        x26=self.conv2d11(x25)
        return x26

m = M().eval()
x21 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x21)
end = time.time()
print(end-start)