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
        self.relu15 = ReLU()
        self.conv2d21 = Conv2d(264, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid3 = Sigmoid()

    def forward(self, x63):
        x64=self.relu15(x63)
        x65=self.conv2d21(x64)
        x66=self.sigmoid3(x65)
        return x66

m = M().eval()
x63 = torch.randn(torch.Size([1, 264, 1, 1]))
start = time.time()
output = m(x63)
end = time.time()
print(end-start)
