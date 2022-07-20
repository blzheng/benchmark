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
        self.relu2 = ReLU(inplace=True)
        self.conv2d3 = Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x9):
        x10=self.relu2(x9)
        x11=self.conv2d3(x10)
        return x11

m = M().eval()
x9 = torch.randn(torch.Size([1, 128, 112, 112]))
start = time.time()
output = m(x9)
end = time.time()
print(end-start)
