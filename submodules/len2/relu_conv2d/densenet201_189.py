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
        self.relu190 = ReLU(inplace=True)
        self.conv2d190 = Conv2d(1760, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x672):
        x673=self.relu190(x672)
        x674=self.conv2d190(x673)
        return x674

m = M().eval()
x672 = torch.randn(torch.Size([1, 1760, 7, 7]))
start = time.time()
output = m(x672)
end = time.time()
print(end-start)
