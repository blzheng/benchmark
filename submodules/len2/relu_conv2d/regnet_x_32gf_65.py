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
        self.relu65 = ReLU(inplace=True)
        self.conv2d69 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x224):
        x225=self.relu65(x224)
        x226=self.conv2d69(x225)
        return x226

m = M().eval()
x224 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
