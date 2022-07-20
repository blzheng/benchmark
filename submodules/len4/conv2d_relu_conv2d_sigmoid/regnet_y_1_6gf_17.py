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
        self.conv2d91 = Conv2d(336, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu71 = ReLU()
        self.conv2d92 = Conv2d(84, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()

    def forward(self, x288):
        x289=self.conv2d91(x288)
        x290=self.relu71(x289)
        x291=self.conv2d92(x290)
        x292=self.sigmoid17(x291)
        return x292

m = M().eval()
x288 = torch.randn(torch.Size([1, 336, 1, 1]))
start = time.time()
output = m(x288)
end = time.time()
print(end-start)
