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
        self.relu19 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.relu20 = ReLU(inplace=True)

    def forward(self, x47):
        x48=self.relu19(x47)
        x49=self.conv2d20(x48)
        x50=self.relu20(x49)
        return x50

m = M().eval()
x47 = torch.randn(torch.Size([1, 64, 27, 27]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
