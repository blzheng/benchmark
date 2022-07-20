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
        self.relu95 = ReLU()
        self.conv2d122 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x385):
        x386=self.relu95(x385)
        x387=self.conv2d122(x386)
        return x387

m = M().eval()
x385 = torch.randn(torch.Size([1, 84, 1, 1]))
start = time.time()
output = m(x385)
end = time.time()
print(end-start)
