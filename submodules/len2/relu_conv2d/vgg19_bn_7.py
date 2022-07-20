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
        self.relu10 = ReLU(inplace=True)
        self.conv2d11 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x35):
        x36=self.relu10(x35)
        x37=self.conv2d11(x36)
        return x37

m = M().eval()
x35 = torch.randn(torch.Size([1, 512, 28, 28]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)