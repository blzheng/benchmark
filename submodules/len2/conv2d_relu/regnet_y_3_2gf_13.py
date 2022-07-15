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
        self.conv2d71 = Conv2d(576, 144, kernel_size=(1, 1), stride=(1, 1))
        self.relu55 = ReLU()

    def forward(self, x224):
        x225=self.conv2d71(x224)
        x226=self.relu55(x225)
        return x226

m = M().eval()
x224 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x224)
end = time.time()
print(end-start)
