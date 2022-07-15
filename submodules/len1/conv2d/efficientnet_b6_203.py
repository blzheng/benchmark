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
        self.conv2d203 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x638):
        x639=self.conv2d203(x638)
        return x639

m = M().eval()
x638 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x638)
end = time.time()
print(end-start)
