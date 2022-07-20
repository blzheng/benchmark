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
        self.conv2d126 = Conv2d(1632, 68, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x392):
        x393=self.conv2d126(x392)
        return x393

m = M().eval()
x392 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x392)
end = time.time()
print(end-start)