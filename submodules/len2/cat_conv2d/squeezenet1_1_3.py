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
        self.conv2d19 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x44, x46):
        x47=torch.cat([x44, x46], 1)
        x48=self.conv2d19(x47)
        return x48

m = M().eval()
x44 = torch.randn(torch.Size([1, 192, 13, 13]))
x46 = torch.randn(torch.Size([1, 192, 13, 13]))
start = time.time()
output = m(x44, x46)
end = time.time()
print(end-start)
