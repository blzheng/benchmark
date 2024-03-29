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
        self.conv2d142 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()

    def forward(self, x455, x452):
        x456=self.conv2d142(x455)
        x457=self.sigmoid23(x456)
        x458=operator.mul(x457, x452)
        return x458

m = M().eval()
x455 = torch.randn(torch.Size([1, 76, 1, 1]))
x452 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x455, x452)
end = time.time()
print(end-start)
