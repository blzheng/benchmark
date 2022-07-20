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
        self.conv2d8 = Conv2d(384, 768, kernel_size=(2, 2), stride=(2, 2))
        self.conv2d9 = Conv2d(768, 768, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), groups=768)

    def forward(self, x83):
        x84=self.conv2d8(x83)
        x86=self.conv2d9(x84)
        return x86

m = M().eval()
x83 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x83)
end = time.time()
print(end-start)
