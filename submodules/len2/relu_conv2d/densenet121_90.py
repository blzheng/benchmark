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
        self.relu91 = ReLU(inplace=True)
        self.conv2d91 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x325):
        x326=self.relu91(x325)
        x327=self.conv2d91(x326)
        return x327

m = M().eval()
x325 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x325)
end = time.time()
print(end-start)
