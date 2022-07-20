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
        self.relu13 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(176, 176, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=176, bias=False)

    def forward(self, x123):
        x124=self.relu13(x123)
        x125=self.conv2d21(x124)
        return x125

m = M().eval()
x123 = torch.randn(torch.Size([1, 176, 14, 14]))
start = time.time()
output = m(x123)
end = time.time()
print(end-start)