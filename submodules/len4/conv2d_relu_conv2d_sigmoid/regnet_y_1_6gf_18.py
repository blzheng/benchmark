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
        self.conv2d96 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu75 = ReLU()
        self.conv2d97 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid18 = Sigmoid()

    def forward(self, x304):
        x305=self.conv2d96(x304)
        x306=self.relu75(x305)
        x307=self.conv2d97(x306)
        x308=self.sigmoid18(x307)
        return x308

m = M().eval()
x304 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x304)
end = time.time()
print(end-start)
