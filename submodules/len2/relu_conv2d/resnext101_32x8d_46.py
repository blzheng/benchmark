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
        self.relu46 = ReLU(inplace=True)
        self.conv2d51 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x165):
        x166=self.relu46(x165)
        x167=self.conv2d51(x166)
        return x167

m = M().eval()
x165 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x165)
end = time.time()
print(end-start)
