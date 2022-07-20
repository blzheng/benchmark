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
        self.relu96 = ReLU(inplace=True)
        self.conv2d124 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x392):
        x393=self.relu96(x392)
        x394=self.conv2d124(x393)
        return x394

m = M().eval()
x392 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x392)
end = time.time()
print(end-start)
