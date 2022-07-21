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
        self.relu12 = ReLU(inplace=True)
        self.conv2d13 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu13 = ReLU(inplace=True)
        self.conv2d14 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu14 = ReLU(inplace=True)
        self.conv2d15 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x29):
        x30=self.relu12(x29)
        x31=self.conv2d13(x30)
        x32=self.relu13(x31)
        x33=self.conv2d14(x32)
        x34=self.relu14(x33)
        x35=self.conv2d15(x34)
        return x35

m = M().eval()
x29 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x29)
end = time.time()
print(end-start)
