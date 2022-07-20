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
        self.relu14 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(168, 168, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x52):
        x53=self.relu14(x52)
        x54=self.conv2d17(x53)
        return x54

m = M().eval()
x52 = torch.randn(torch.Size([1, 168, 28, 28]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)
