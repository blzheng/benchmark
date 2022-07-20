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
        self.relu21 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)

    def forward(self, x102):
        x103=self.relu21(x102)
        x104=self.conv2d32(x103)
        return x104

m = M().eval()
x102 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
