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
        self.conv2d84 = Conv2d(480, 20, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x253):
        x254=x253.mean((2, 3),keepdim=True)
        x255=self.conv2d84(x254)
        return x255

m = M().eval()
x253 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
