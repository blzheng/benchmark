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
        self.relu97 = ReLU(inplace=True)
        self.conv2d102 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x334):
        x335=self.relu97(x334)
        x336=self.conv2d102(x335)
        return x336

m = M().eval()
x334 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x334)
end = time.time()
print(end-start)
