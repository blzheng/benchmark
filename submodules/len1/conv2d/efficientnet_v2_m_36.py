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
        self.conv2d36 = Conv2d(640, 40, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x121):
        x122=self.conv2d36(x121)
        return x122

m = M().eval()
x121 = torch.randn(torch.Size([1, 640, 1, 1]))
start = time.time()
output = m(x121)
end = time.time()
print(end-start)
