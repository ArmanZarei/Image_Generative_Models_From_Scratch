import torch


class AdjustPixelsRange(object):
    def __init__(self, range_in, range_out, device='cpu'):
        self._range_in = torch.tensor(range_in, dtype=torch.float32).to(device)
        self._range_out = torch.tensor(range_out, dtype=torch.float32).to(device)

        if range_in == range_out:
            self.scale = 1
            self.bias = 0
        else:
            self.scale = (self._range_out[1] - self._range_out[0]) / (self._range_in[1] - self._range_in[0])
            self.bias = self._range_out[0] - self.scale * self._range_in[0]
    
    def __call__(self, x):
        return torch.clamp(self.scale * x + self.bias, min=self._range_out[0], max=self._range_out[1])