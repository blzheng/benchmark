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
        self.conv2d9 = Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1))
        self.relu7 = ReLU()
        self.conv2d10 = Conv2d(12, 48, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x28):
        x29=self.conv2d9(x28)
        x30=self.relu7(x29)
        x31=self.conv2d10(x30)
        return x31

m = M().eval()
x28 = torch.randn(torch.Size([1, 48, 1, 1]))
start = time.time()
output = m(x28)
end = time.time()
print(end-start)
