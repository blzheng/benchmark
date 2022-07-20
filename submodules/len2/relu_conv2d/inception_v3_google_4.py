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
        self.conv2d9 = Conv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x41):
        x42=torch.nn.functional.relu(x41,inplace=True)
        x43=self.conv2d9(x42)
        return x43

m = M().eval()
x41 = torch.randn(torch.Size([1, 64, 25, 25]))
start = time.time()
output = m(x41)
end = time.time()
print(end-start)
