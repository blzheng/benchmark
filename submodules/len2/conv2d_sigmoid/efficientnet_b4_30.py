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
        self.conv2d152 = Conv2d(68, 1632, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid30 = Sigmoid()

    def forward(self, x474):
        x475=self.conv2d152(x474)
        x476=self.sigmoid30(x475)
        return x476

m = M().eval()
x474 = torch.randn(torch.Size([1, 68, 1, 1]))
start = time.time()
output = m(x474)
end = time.time()
print(end-start)
