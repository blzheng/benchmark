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
        self.relu7 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x18):
        x19=self.relu7(x18)
        x20=self.conv2d8(x19)
        return x20

m = M().eval()
x18 = torch.randn(torch.Size([1, 32, 54, 54]))
start = time.time()
output = m(x18)
end = time.time()
print(end-start)
