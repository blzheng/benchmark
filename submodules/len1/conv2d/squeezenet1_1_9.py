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
        self.conv2d9 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x20):
        x23=self.conv2d9(x20)
        return x23

m = M().eval()
x20 = torch.randn(torch.Size([1, 32, 27, 27]))
start = time.time()
output = m(x20)
end = time.time()
print(end-start)
