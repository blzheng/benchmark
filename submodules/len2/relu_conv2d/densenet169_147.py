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
        self.relu148 = ReLU(inplace=True)
        self.conv2d148 = Conv2d(1344, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x525):
        x526=self.relu148(x525)
        x527=self.conv2d148(x526)
        return x527

m = M().eval()
x525 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x525)
end = time.time()
print(end-start)
