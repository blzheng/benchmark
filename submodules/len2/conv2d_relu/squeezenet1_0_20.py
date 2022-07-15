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
        self.conv2d20 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.relu20 = ReLU(inplace=True)

    def forward(self, x48):
        x49=self.conv2d20(x48)
        x50=self.relu20(x49)
        return x50

m = M().eval()
x48 = torch.randn(torch.Size([1, 64, 27, 27]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
