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
        self.conv2d17 = Conv2d(8, 192, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x50):
        x51=self.conv2d17(x50)
        return x51

m = M().eval()
x50 = torch.randn(torch.Size([1, 8, 1, 1]))
start = time.time()
output = m(x50)
end = time.time()
print(end-start)
