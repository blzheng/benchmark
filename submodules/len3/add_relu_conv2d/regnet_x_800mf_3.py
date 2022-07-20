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
        self.relu12 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(128, 288, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x37, x45):
        x46=operator.add(x37, x45)
        x47=self.relu12(x46)
        x48=self.conv2d15(x47)
        return x48

m = M().eval()
x37 = torch.randn(torch.Size([1, 128, 28, 28]))
x45 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x37, x45)
end = time.time()
print(end-start)