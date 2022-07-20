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
        self.conv2d62 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x190):
        x191=self.conv2d62(x190)
        x192=self.sigmoid12(x191)
        return x192

m = M().eval()
x190 = torch.randn(torch.Size([1, 20, 1, 1]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
