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
        self.relu51 = ReLU()
        self.conv2d67 = Conv2d(144, 576, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x209):
        x210=self.relu51(x209)
        x211=self.conv2d67(x210)
        return x211

m = M().eval()
x209 = torch.randn(torch.Size([1, 144, 1, 1]))
start = time.time()
output = m(x209)
end = time.time()
print(end-start)
