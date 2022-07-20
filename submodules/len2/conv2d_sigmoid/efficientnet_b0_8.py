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
        self.conv2d43 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x129):
        x130=self.conv2d43(x129)
        x131=self.sigmoid8(x130)
        return x131

m = M().eval()
x129 = torch.randn(torch.Size([1, 20, 1, 1]))
start = time.time()
output = m(x129)
end = time.time()
print(end-start)
