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
        self.relu63 = ReLU()
        self.conv2d82 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x257):
        x258=self.relu63(x257)
        x259=self.conv2d82(x258)
        return x259

m = M().eval()
x257 = torch.randn(torch.Size([1, 726, 1, 1]))
start = time.time()
output = m(x257)
end = time.time()
print(end-start)
