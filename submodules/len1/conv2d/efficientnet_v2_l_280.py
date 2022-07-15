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
        self.conv2d280 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x902):
        x903=self.conv2d280(x902)
        return x903

m = M().eval()
x902 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x902)
end = time.time()
print(end-start)
