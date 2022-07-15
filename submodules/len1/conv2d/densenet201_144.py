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
        self.conv2d144 = Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x512):
        x513=self.conv2d144(x512)
        return x513

m = M().eval()
x512 = torch.randn(torch.Size([1, 1024, 7, 7]))
start = time.time()
output = m(x512)
end = time.time()
print(end-start)
