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
        self.conv2d92 = Conv2d(34, 816, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x284):
        x285=self.conv2d92(x284)
        x286=self.sigmoid18(x285)
        return x286

m = M().eval()
x284 = torch.randn(torch.Size([1, 34, 1, 1]))
start = time.time()
output = m(x284)
end = time.time()
print(end-start)
