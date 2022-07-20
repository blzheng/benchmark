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
        self.batchnorm2d71 = BatchNorm2d(768, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu71 = ReLU(inplace=True)
        self.conv2d71 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x253):
        x254=self.batchnorm2d71(x253)
        x255=self.relu71(x254)
        x256=self.conv2d71(x255)
        return x256

m = M().eval()
x253 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
