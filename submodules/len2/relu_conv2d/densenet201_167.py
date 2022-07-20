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
        self.relu168 = ReLU(inplace=True)
        self.conv2d168 = Conv2d(1408, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x595):
        x596=self.relu168(x595)
        x597=self.conv2d168(x596)
        return x597

m = M().eval()
x595 = torch.randn(torch.Size([1, 1408, 7, 7]))
start = time.time()
output = m(x595)
end = time.time()
print(end-start)
