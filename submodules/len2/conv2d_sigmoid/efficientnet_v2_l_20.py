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
        self.conv2d136 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x442):
        x443=self.conv2d136(x442)
        x444=self.sigmoid20(x443)
        return x444

m = M().eval()
x442 = torch.randn(torch.Size([1, 56, 1, 1]))
start = time.time()
output = m(x442)
end = time.time()
print(end-start)
