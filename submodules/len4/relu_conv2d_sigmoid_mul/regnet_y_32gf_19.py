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
        self.conv2d103 = Conv2d(348, 3712, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()

    def forward(self, x323, x321):
        x324=self.relu79(x323)
        x325=self.conv2d103(x324)
        x326=self.sigmoid19(x325)
        x327=operator.mul(x326, x321)
        return x327

m = M().eval()
x323 = torch.randn(torch.Size([1, 348, 1, 1]))
x321 = torch.randn(torch.Size([1, 3712, 7, 7]))
start = time.time()
output = m(x323, x321)
end = time.time()
print(end-start)
