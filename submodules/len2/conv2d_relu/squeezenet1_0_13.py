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
        self.conv2d13 = Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
        self.relu13 = ReLU(inplace=True)

    def forward(self, x32):
        x33=self.conv2d13(x32)
        x34=self.relu13(x33)
        return x34

m = M().eval()
x32 = torch.randn(torch.Size([1, 256, 27, 27]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
