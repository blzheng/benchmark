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
        self.conv2d152 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid25 = Sigmoid()

    def forward(self, x487):
        x488=self.conv2d152(x487)
        x489=self.sigmoid25(x488)
        return x489

m = M().eval()
x487 = torch.randn(torch.Size([1, 76, 1, 1]))
start = time.time()
output = m(x487)
end = time.time()
print(end-start)
