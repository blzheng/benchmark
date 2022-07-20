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
        self.relu10 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x44, x46):
        x47=operator.add(x44, x46)
        x48=self.relu10(x47)
        x49=self.conv2d15(x48)
        return x49

m = M().eval()
x44 = torch.randn(torch.Size([1, 512, 28, 28]))
x46 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x44, x46)
end = time.time()
print(end-start)
