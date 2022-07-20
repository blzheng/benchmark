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
        self.conv2d211 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid35 = Sigmoid()

    def forward(self, x680):
        x681=self.conv2d211(x680)
        x682=self.sigmoid35(x681)
        return x682

m = M().eval()
x680 = torch.randn(torch.Size([1, 96, 1, 1]))
start = time.time()
output = m(x680)
end = time.time()
print(end-start)
