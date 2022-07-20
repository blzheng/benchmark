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
        self.relu53 = ReLU(inplace=True)
        self.conv2d58 = Conv2d(912, 912, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x186):
        x187=self.relu53(x186)
        x188=self.conv2d58(x187)
        return x188

m = M().eval()
x186 = torch.randn(torch.Size([1, 912, 7, 7]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)
