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
        self.conv2d219 = Conv2d(2304, 96, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x655):
        x656=x655.mean((2, 3),keepdim=True)
        x657=self.conv2d219(x656)
        return x657

m = M().eval()
x655 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x655)
end = time.time()
print(end-start)
