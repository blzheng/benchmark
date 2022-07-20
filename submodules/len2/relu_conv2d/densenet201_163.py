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
        self.relu164 = ReLU(inplace=True)
        self.conv2d164 = Conv2d(1344, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x581):
        x582=self.relu164(x581)
        x583=self.conv2d164(x582)
        return x583

m = M().eval()
x581 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x581)
end = time.time()
print(end-start)
