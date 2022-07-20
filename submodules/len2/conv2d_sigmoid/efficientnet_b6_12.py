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
        self.conv2d61 = Conv2d(18, 432, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x189):
        x190=self.conv2d61(x189)
        x191=self.sigmoid12(x190)
        return x191

m = M().eval()
x189 = torch.randn(torch.Size([1, 18, 1, 1]))
start = time.time()
output = m(x189)
end = time.time()
print(end-start)
