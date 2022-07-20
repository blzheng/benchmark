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
        self.relu107 = ReLU()
        self.conv2d138 = Conv2d(726, 7392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid26 = Sigmoid()

    def forward(self, x435):
        x436=self.relu107(x435)
        x437=self.conv2d138(x436)
        x438=self.sigmoid26(x437)
        return x438

m = M().eval()
x435 = torch.randn(torch.Size([1, 726, 1, 1]))
start = time.time()
output = m(x435)
end = time.time()
print(end-start)
