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
        self.relu7 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x37):
        x38=self.relu7(x37)
        x39=self.conv2d11(x38)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x37)
end = time.time()
print(end-start)
