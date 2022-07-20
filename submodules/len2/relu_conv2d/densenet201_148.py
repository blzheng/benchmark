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
        self.relu149 = ReLU(inplace=True)
        self.conv2d149 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x528):
        x529=self.relu149(x528)
        x530=self.conv2d149(x529)
        return x530

m = M().eval()
x528 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x528)
end = time.time()
print(end-start)
