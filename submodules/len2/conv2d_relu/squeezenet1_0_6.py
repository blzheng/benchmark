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
        self.conv2d6 = Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu6 = ReLU(inplace=True)

    def forward(self, x12):
        x15=self.conv2d6(x12)
        x16=self.relu6(x15)
        return x16

m = M().eval()
x12 = torch.randn(torch.Size([1, 16, 54, 54]))
start = time.time()
output = m(x12)
end = time.time()
print(end-start)