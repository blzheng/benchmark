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
        self.relu88 = ReLU(inplace=True)
        self.conv2d92 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x302):
        x303=self.relu88(x302)
        x304=self.conv2d92(x303)
        return x304

m = M().eval()
x302 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x302)
end = time.time()
print(end-start)
