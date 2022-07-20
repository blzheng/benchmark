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
        self.relu71 = ReLU()
        self.conv2d92 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x289):
        x290=self.relu71(x289)
        x291=self.conv2d92(x290)
        return x291

m = M().eval()
x289 = torch.randn(torch.Size([1, 726, 1, 1]))
start = time.time()
output = m(x289)
end = time.time()
print(end-start)
