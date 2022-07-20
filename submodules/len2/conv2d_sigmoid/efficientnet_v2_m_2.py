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
        self.conv2d37 = Conv2d(40, 640, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()

    def forward(self, x123):
        x124=self.conv2d37(x123)
        x125=self.sigmoid2(x124)
        return x125

m = M().eval()
x123 = torch.randn(torch.Size([1, 40, 1, 1]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)
