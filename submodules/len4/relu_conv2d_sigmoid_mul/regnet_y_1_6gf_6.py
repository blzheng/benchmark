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
        self.relu27 = ReLU()
        self.conv2d36 = Conv2d(30, 120, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()

    def forward(self, x111, x109):
        x112=self.relu27(x111)
        x113=self.conv2d36(x112)
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        return x115

m = M().eval()
x111 = torch.randn(torch.Size([1, 30, 1, 1]))
x109 = torch.randn(torch.Size([1, 120, 28, 28]))
start = time.time()
output = m(x111, x109)
end = time.time()
print(end-start)
