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
        self.conv2d42 = Conv2d(14, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x128, x125):
        x129=self.conv2d42(x128)
        x130=self.sigmoid8(x129)
        x131=operator.mul(x130, x125)
        return x131

m = M().eval()
x128 = torch.randn(torch.Size([1, 14, 1, 1]))
x125 = torch.randn(torch.Size([1, 336, 28, 28]))
start = time.time()
output = m(x128, x125)
end = time.time()
print(end-start)
