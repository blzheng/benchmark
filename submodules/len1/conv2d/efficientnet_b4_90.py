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
        self.conv2d90 = Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)

    def forward(self, x278):
        x279=self.conv2d90(x278)
        return x279

m = M().eval()
x278 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)