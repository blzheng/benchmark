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
        self.conv2d21 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x49):
        x52=self.conv2d21(x49)
        return x52

m = M().eval()
x49 = torch.randn(torch.Size([1, 64, 13, 13]))
start = time.time()
output = m(x49)
end = time.time()
print(end-start)
