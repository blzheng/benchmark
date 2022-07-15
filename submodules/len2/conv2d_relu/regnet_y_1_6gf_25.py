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
        self.conv2d132 = Conv2d(888, 84, kernel_size=(1, 1), stride=(1, 1))
        self.relu103 = ReLU()

    def forward(self, x418):
        x419=self.conv2d132(x418)
        x420=self.relu103(x419)
        return x420

m = M().eval()
x418 = torch.randn(torch.Size([1, 888, 1, 1]))
start = time.time()
output = m(x418)
end = time.time()
print(end-start)
