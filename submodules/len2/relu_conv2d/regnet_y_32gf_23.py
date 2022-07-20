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
        self.relu31 = ReLU()
        self.conv2d42 = Conv2d(174, 1392, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x129):
        x130=self.relu31(x129)
        x131=self.conv2d42(x130)
        return x131

m = M().eval()
x129 = torch.randn(torch.Size([1, 174, 1, 1]))
start = time.time()
output = m(x129)
end = time.time()
print(end-start)
