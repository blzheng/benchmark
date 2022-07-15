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
        self.conv2d115 = Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x344):
        x345=self.conv2d115(x344)
        return x345

m = M().eval()
x344 = torch.randn(torch.Size([1, 1056, 1, 1]))
start = time.time()
output = m(x344)
end = time.time()
print(end-start)
