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
        self.conv2d19 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU(inplace=True)
        self.conv2d20 = Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        self.relu20 = ReLU(inplace=True)

    def forward(self, x47):
        x48=self.conv2d19(x47)
        x49=self.relu19(x48)
        x50=self.conv2d20(x49)
        x51=self.relu20(x50)
        return x51

m = M().eval()
x47 = torch.randn(torch.Size([1, 384, 13, 13]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
