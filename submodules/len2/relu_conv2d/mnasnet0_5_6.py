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
        self.relu6 = ReLU(inplace=True)
        self.conv2d10 = Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)

    def forward(self, x27):
        x28=self.relu6(x27)
        x29=self.conv2d10(x28)
        return x29

m = M().eval()
x27 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x27)
end = time.time()
print(end-start)
