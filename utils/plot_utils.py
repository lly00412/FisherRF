'''
refer to https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/image_utils.py
'''
import torch
import matplotlib.pyplot as plt
def colormap(map, cmap="turbo",max=None, min=None):
    if max==None:
        max = map.max()
    if min==None:
        min = map.min()
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map[map>max] = max
    map[map<min] = min
    map = (map - min) / (max - min)
    map = (map * 255).round().long().squeeze()
    map = colors[map].permute(2,0,1)
    return map