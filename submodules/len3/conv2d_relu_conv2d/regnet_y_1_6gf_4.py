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
        self.conv2d25 = Conv2d(120, 30, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU()
        self.conv2d26 = Conv2d(30, 120, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x78):
        x79=self.conv2d25(x78)
        x80=self.relu19(x79)
        x81=self.conv2d26(x80)
        return x81

m = M().eval()
x78 = torch.randn(torch.Size([1, 120, 1, 1]))
start = time.time()
output = m(x78)
end = time.time()
print(end-start)
