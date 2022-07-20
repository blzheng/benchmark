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
        self.relu153 = ReLU(inplace=True)
        self.conv2d153 = Conv2d(192, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x542):
        x543=self.relu153(x542)
        x544=self.conv2d153(x543)
        return x544

m = M().eval()
x542 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x542)
end = time.time()
print(end-start)
