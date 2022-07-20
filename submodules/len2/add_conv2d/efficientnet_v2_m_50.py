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
        self.conv2d244 = Conv2d(512, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x779, x764):
        x780=operator.add(x779, x764)
        x781=self.conv2d244(x780)
        return x781

m = M().eval()
x779 = torch.randn(torch.Size([1, 512, 7, 7]))
x764 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x779, x764)
end = time.time()
print(end-start)
