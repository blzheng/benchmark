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
        self.conv2d4 = Conv2d(96, 192, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d5 = Conv2d(192, 192, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=192)

    def forward(self, x44):
        x45=self.conv2d4(x44)
        x47=self.conv2d5(x45)
        return x47

m = M().eval()
x44 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x44)
end = time.time()
print(end-start)
