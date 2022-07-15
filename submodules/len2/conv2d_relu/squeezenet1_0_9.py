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
        self.conv2d9 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu9 = ReLU(inplace=True)

    def forward(self, x19):
        x22=self.conv2d9(x19)
        x23=self.relu9(x22)
        return x23

m = M().eval()
x19 = torch.randn(torch.Size([1, 32, 54, 54]))
start = time.time()
output = m(x19)
end = time.time()
print(end-start)
