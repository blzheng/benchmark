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
        self.conv2d26 = Conv2d(320, 36, kernel_size=(1, 1), stride=(1, 1))
        self.relu19 = ReLU()
        self.conv2d27 = Conv2d(36, 320, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x80):
        x81=self.conv2d26(x80)
        x82=self.relu19(x81)
        x83=self.conv2d27(x82)
        return x83

m = M().eval()
x80 = torch.randn(torch.Size([1, 320, 1, 1]))
start = time.time()
output = m(x80)
end = time.time()
print(end-start)
