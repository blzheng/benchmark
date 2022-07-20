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
        self.conv2d1 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x1):
        x2=self.relu0(x1)
        x3=self.conv2d1(x2)
        return x3

m = M().eval()
x1 = torch.randn(torch.Size([1, 64, 224, 224]))
start = time.time()
output = m(x1)
end = time.time()
print(end-start)
