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
        self.conv2d35 = Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=20, bias=False)

    def forward(self, x108):
        x109=self.conv2d35(x108)
        return x109

m = M().eval()
x108 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x108)
end = time.time()
print(end-start)
