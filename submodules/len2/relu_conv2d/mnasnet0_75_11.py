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
        self.relu11 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(96, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x47):
        x48=self.relu11(x47)
        x49=self.conv2d17(x48)
        return x49

m = M().eval()
x47 = torch.randn(torch.Size([1, 96, 28, 28]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
