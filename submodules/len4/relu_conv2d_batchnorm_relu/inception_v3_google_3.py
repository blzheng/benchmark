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
        self.conv2d7 = Conv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.batchnorm2d7 = BatchNorm2d(64, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x35):
        x36=torch.nn.functional.relu(x35,inplace=True)
        x37=self.conv2d7(x36)
        x38=self.batchnorm2d7(x37)
        x39=torch.nn.functional.relu(x38,inplace=True)
        return x39

m = M().eval()
x35 = torch.randn(torch.Size([1, 48, 25, 25]))
start = time.time()
output = m(x35)
end = time.time()
print(end-start)
