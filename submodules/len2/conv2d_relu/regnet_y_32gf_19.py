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
        self.conv2d102 = Conv2d(3712, 348, kernel_size=(1, 1), stride=(1, 1))
        self.relu79 = ReLU()

    def forward(self, x322):
        x323=self.conv2d102(x322)
        x324=self.relu79(x323)
        return x324

m = M().eval()
x322 = torch.randn(torch.Size([1, 3712, 1, 1]))
start = time.time()
output = m(x322)
end = time.time()
print(end-start)
