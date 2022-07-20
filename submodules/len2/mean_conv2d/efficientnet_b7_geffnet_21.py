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
        self.conv2d104 = Conv2d(960, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x312):
        x313=x312.mean((2, 3),keepdim=True)
        x314=self.conv2d104(x313)
        return x314

m = M().eval()
x312 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x312)
end = time.time()
print(end-start)
