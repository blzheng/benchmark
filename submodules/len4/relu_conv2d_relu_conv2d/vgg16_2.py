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
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu11 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x25):
        x26=self.relu10(x25)
        x27=self.conv2d11(x26)
        x28=self.relu11(x27)
        x29=self.conv2d12(x28)
        return x29

m = M().eval()
x25 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
