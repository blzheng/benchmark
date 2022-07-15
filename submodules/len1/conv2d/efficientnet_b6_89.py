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
        self.conv2d89 = Conv2d(864, 864, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=864, bias=False)

    def forward(self, x277):
        x278=self.conv2d89(x277)
        return x278

m = M().eval()
x277 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x277)
end = time.time()
print(end-start)
