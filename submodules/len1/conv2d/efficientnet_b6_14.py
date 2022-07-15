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
        self.conv2d14 = Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)

    def forward(self, x43):
        x44=self.conv2d14(x43)
        return x44

m = M().eval()
x43 = torch.randn(torch.Size([1, 192, 112, 112]))
start = time.time()
output = m(x43)
end = time.time()
print(end-start)
