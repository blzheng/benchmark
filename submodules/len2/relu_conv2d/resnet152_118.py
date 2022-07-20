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
        self.relu118 = ReLU(inplace=True)
        self.conv2d123 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x405):
        x406=self.relu118(x405)
        x407=self.conv2d123(x406)
        return x407

m = M().eval()
x405 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x405)
end = time.time()
print(end-start)
