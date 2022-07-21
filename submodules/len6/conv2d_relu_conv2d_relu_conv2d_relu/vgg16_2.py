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
        self.conv2d10 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu11 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu12 = ReLU(inplace=True)

    def forward(self, x24):
        x25=self.conv2d10(x24)
        x26=self.relu10(x25)
        x27=self.conv2d11(x26)
        x28=self.relu11(x27)
        x29=self.conv2d12(x28)
        x30=self.relu12(x29)
        return x30

m = M().eval()
x24 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
