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
        self.relu173 = ReLU(inplace=True)
        self.conv2d173 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x612):
        x613=self.relu173(x612)
        x614=self.conv2d173(x613)
        return x614

m = M().eval()
x612 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x612)
end = time.time()
print(end-start)
