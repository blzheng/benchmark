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
        self.conv2d180 = Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x565):
        x566=self.conv2d180(x565)
        return x566

m = M().eval()
x565 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x565)
end = time.time()
print(end-start)
