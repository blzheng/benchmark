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
        self.relu155 = ReLU(inplace=True)
        self.conv2d155 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x549):
        x550=self.relu155(x549)
        x551=self.conv2d155(x550)
        return x551

m = M().eval()
x549 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x549)
end = time.time()
print(end-start)
