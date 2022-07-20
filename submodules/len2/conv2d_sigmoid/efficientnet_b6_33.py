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
        self.conv2d166 = Conv2d(86, 2064, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid33 = Sigmoid()

    def forward(self, x519):
        x520=self.conv2d166(x519)
        x521=self.sigmoid33(x520)
        return x521

m = M().eval()
x519 = torch.randn(torch.Size([1, 86, 1, 1]))
start = time.time()
output = m(x519)
end = time.time()
print(end-start)
