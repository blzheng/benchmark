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
        self.relu0 = ReLU(inplace=True)
        self.conv2d1 = Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)

    def forward(self, x2):
        x3=self.relu0(x2)
        x4=self.conv2d1(x3)
        return x4

m = M().eval()
x2 = torch.randn(torch.Size([1, 16, 112, 112]))
start = time.time()
output = m(x2)
end = time.time()
print(end-start)
