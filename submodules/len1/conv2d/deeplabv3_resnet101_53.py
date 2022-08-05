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
        self.conv2d53 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)

    def forward(self, x175):
        x176=self.conv2d53(x175)
        return x176

m = M().eval()
x175 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
