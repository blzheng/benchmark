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
        self.batchnorm2d53 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu49 = ReLU(inplace=True)
        self.conv2d54 = Conv2d(400, 400, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=25, bias=False)
        self.batchnorm2d54 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x172):
        x173=self.batchnorm2d53(x172)
        x174=self.relu49(x173)
        x175=self.conv2d54(x174)
        x176=self.batchnorm2d54(x175)
        return x176

m = M().eval()
x172 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x172)
end = time.time()
print(end-start)