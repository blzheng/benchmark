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
        self.relu140 = ReLU(inplace=True)
        self.conv2d140 = Conv2d(960, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x497):
        x498=self.relu140(x497)
        x499=self.conv2d140(x498)
        return x499

m = M().eval()
x497 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x497)
end = time.time()
print(end-start)
