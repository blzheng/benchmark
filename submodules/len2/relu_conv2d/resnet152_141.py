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
        self.relu142 = ReLU(inplace=True)
        self.conv2d146 = Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x482):
        x483=self.relu142(x482)
        x484=self.conv2d146(x483)
        return x484

m = M().eval()
x482 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x482)
end = time.time()
print(end-start)
