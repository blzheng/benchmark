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
        self.conv2d14 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu14 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu15 = ReLU(inplace=True)

    def forward(self, x32):
        x33=self.conv2d14(x32)
        x34=self.relu14(x33)
        x35=self.conv2d15(x34)
        x36=self.relu15(x35)
        return x36

m = M().eval()
x32 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x32)
end = time.time()
print(end-start)
