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
        self.conv2d15 = Conv2d(8, 32, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x47):
        x48=self.conv2d15(x47)
        return x48

m = M().eval()
x47 = torch.randn(torch.Size([1, 8, 1, 1]))
start = time.time()
output = m(x47)
end = time.time()
print(end-start)
