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
        self.relu124 = ReLU(inplace=True)
        self.conv2d130 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x428, x420):
        x429=operator.add(x428, x420)
        x430=self.relu124(x429)
        x431=self.conv2d130(x430)
        return x431

m = M().eval()
x428 = torch.randn(torch.Size([1, 1024, 14, 14]))
x420 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x428, x420)
end = time.time()
print(end-start)
