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
        self.relu33 = ReLU(inplace=True)
        self.conv2d38 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x111, x119):
        x120=operator.add(x111, x119)
        x121=self.relu33(x120)
        x122=self.conv2d38(x121)
        return x122

m = M().eval()
x111 = torch.randn(torch.Size([1, 400, 7, 7]))
x119 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x111, x119)
end = time.time()
print(end-start)
