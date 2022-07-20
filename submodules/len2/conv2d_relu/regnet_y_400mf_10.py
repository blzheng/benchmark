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
        self.conv2d57 = Conv2d(440, 52, kernel_size=(1, 1), stride=(1, 1))
        self.relu43 = ReLU()

    def forward(self, x178):
        x179=self.conv2d57(x178)
        x180=self.relu43(x179)
        return x180

m = M().eval()
x178 = torch.randn(torch.Size([1, 440, 1, 1]))
start = time.time()
output = m(x178)
end = time.time()
print(end-start)