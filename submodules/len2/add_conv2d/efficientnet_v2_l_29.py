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
        self.conv2d113 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x368, x353):
        x369=operator.add(x368, x353)
        x370=self.conv2d113(x369)
        return x370

m = M().eval()
x368 = torch.randn(torch.Size([1, 224, 14, 14]))
x353 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x368, x353)
end = time.time()
print(end-start)