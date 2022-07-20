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
        self.conv2d12 = Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu12 = ReLU(inplace=True)

    def forward(self, x27):
        x30=self.conv2d12(x27)
        x31=self.relu12(x30)
        return x31

m = M().eval()
x27 = torch.randn(torch.Size([1, 32, 27, 27]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)