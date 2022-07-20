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
        self.relu79 = ReLU()
        self.conv2d102 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x321):
        x322=self.relu79(x321)
        x323=self.conv2d102(x322)
        return x323

m = M().eval()
x321 = torch.randn(torch.Size([1, 726, 1, 1]))
start = time.time()
output = m(x321)
end = time.time()
print(end-start)
