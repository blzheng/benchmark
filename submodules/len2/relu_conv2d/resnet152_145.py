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
        self.relu145 = ReLU(inplace=True)
        self.conv2d151 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x497):
        x498=self.relu145(x497)
        x499=self.conv2d151(x498)
        return x499

m = M().eval()
x497 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x497)
end = time.time()
print(end-start)
