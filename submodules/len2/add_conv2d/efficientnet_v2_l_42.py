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
        self.conv2d178 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x576, x561):
        x577=operator.add(x576, x561)
        x578=self.conv2d178(x577)
        return x578

m = M().eval()
x576 = torch.randn(torch.Size([1, 224, 14, 14]))
x561 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x576, x561)
end = time.time()
print(end-start)
