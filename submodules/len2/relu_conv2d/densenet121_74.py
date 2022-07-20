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
        self.relu75 = ReLU(inplace=True)
        self.conv2d75 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x268):
        x269=self.relu75(x268)
        x270=self.conv2d75(x269)
        return x270

m = M().eval()
x268 = torch.randn(torch.Size([1, 832, 14, 14]))
start = time.time()
output = m(x268)
end = time.time()
print(end-start)
