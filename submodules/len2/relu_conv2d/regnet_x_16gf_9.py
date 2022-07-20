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
        self.relu9 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x36):
        x37=self.relu9(x36)
        x38=self.conv2d12(x37)
        return x38

m = M().eval()
x36 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x36)
end = time.time()
print(end-start)
