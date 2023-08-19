import torch

torch.set_default_dtype(torch.float64)

lim_val_exp = torch.log(torch.nextafter(torch.tensor(float('inf')), torch.tensor(1.0)))
lim_val_log = torch.nextafter(torch.tensor(0.0), torch.tensor(1.0))


def safe_exp(x): 
    clamp_x = torch.clamp(x, min=-torch.inf, max=lim_val_exp)
    return torch.exp(clamp_x) # 出力の上限値が 1.7977e+308 に制限される


def safe_log(x):
    clamp_x = torch.clamp(x, min=lim_val_log, max=torch.inf)
    return torch.log(clamp_x) # 出力の下限値が -744.4401 に制限される


def safe_sqrt(x):
    clamp_x = torch.clamp(x, min=torch.tensor(0.0), max=torch.inf)
    return torch.sqrt(clamp_x)