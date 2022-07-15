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
        self.conv2d265 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x792):
        x793=self.conv2d265(x792)
        return x793

m = M().eval()
x792 = torch.randn(torch.Size([1, 160, 1, 1]))
start = time.time()
output = m(x792)
end = time.time()
print(end-start)
