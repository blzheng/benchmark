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
        self.conv2d156 = Conv2d(50, 1200, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid31 = Sigmoid()

    def forward(self, x489):
        x490=self.conv2d156(x489)
        x491=self.sigmoid31(x490)
        return x491

m = M().eval()
x489 = torch.randn(torch.Size([1, 50, 1, 1]))
start = time.time()
output = m(x489)
end = time.time()
print(end-start)
