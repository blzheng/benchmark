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
        self.conv2d114 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x342):
        x343=x342.mean((2, 3),keepdim=True)
        x344=self.conv2d114(x343)
        return x344

m = M().eval()
x342 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x342)
end = time.time()
print(end-start)
