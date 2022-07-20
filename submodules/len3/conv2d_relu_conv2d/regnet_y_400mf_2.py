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
        self.conv2d15 = Conv2d(104, 26, kernel_size=(1, 1), stride=(1, 1))
        self.relu11 = ReLU()
        self.conv2d16 = Conv2d(26, 104, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x46):
        x47=self.conv2d15(x46)
        x48=self.relu11(x47)
        x49=self.conv2d16(x48)
        return x49

m = M().eval()
x46 = torch.randn(torch.Size([1, 104, 1, 1]))
start = time.time()
output = m(x46)
end = time.time()
print(end-start)
