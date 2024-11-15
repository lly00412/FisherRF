import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

import torch

from utils.plot_utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_unmap(fname,rgb_std,mask,q=0.6):
    fig, ax = plt.subplots(1)
    percentail = torch.quantile(rgb_std[mask], q=q)
    rgb_std_clipped = torch.clip(rgb_std, max=percentail)
    data_min = 0.0
    data_max = percentail.cpu().numpy()
    plt.figure(facecolor='white')
    heatmap = sns.heatmap(rgb_std_clipped.detach().cpu(), square=True, mask=~mask.detach().cpu().numpy(), cbar=False, cmap="viridis")
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(fname)
    plt.close()

if __name__ == '__main__':

    # m360 garden, blender lego,  T&T playroom

    # output_path = "./output/"
    # dataset = 'nerf_synthetic'
    # scene = 'lego'
    # load_path = os.path.join(output_path, dataset,scene, 'v15')
    # fname = 'r_125'
    # idx = '125'
    # u_file = os.path.join(load_path, 'eval_seed_0',f'uncertainty_{idx}_{fname}.npz')
    # u_dict = np.load(u_file,allow_pickle=True)

    # fisherRF = u_dict['uncertanity_map']
    # mask = (fisherRF>0.)
    # save_name = f'fisherRF_{fname}.jpg'
    # plot_unmap(load_path, torch.from_numpy(fisherRF), torch.from_numpy(mask), save_name, q=0.6)
    #
    # rests = u_dict['rests'].tolist()
    # vcurf = rests['vcu(6 vcams, 0.1 med)']
    # vcurf[vcurf==2] = 0.
    # mask = (vcurf > 0.)
    # save_name = f'vcurf_{fname}.jpg'
    # plot_unmap(load_path, vcurf, mask, save_name, q=0.6)
    #
    # err = rests['rgb_err']
    # mask = (err>0)
    # save_name = f'err_{fname}.jpg'
    # plot_unmap(load_path, err, mask, save_name, q=0.6)
    #
    # load_path = os.path.join(output_path, dataset, scene)
    # u_file = os.path.join(load_path, 'emsemble_eval',f'uncertainty_{idx}_{fname}.npz')
    # u_dict = np.load(u_file,allow_pickle=True)
    #
    # save_path = os.path.join(load_path, 'v15')
    # depth_std = u_dict['depth_std']
    # mask = (depth_std>0.)
    # save_name = f'emsemble_{fname}_depth_std.jpg'
    # plot_unmap(save_path, torch.from_numpy(depth_std), torch.from_numpy(mask), save_name, q=0.6)
    #
    # rgb_std = u_dict['rgb_std']
    # mask = (rgb_std > 0.)
    # save_name = f'emsemble_{fname}_rgb_std.jpg'
    # plot_unmap(save_path, torch.from_numpy(rgb_std), torch.from_numpy(mask), save_name, q=0.6)

    # for activenerf

    # output_path = "./output/"
    # dataset = 'nerf_synthetic'
    # scene = 'lego'
    # load_path = os.path.join(output_path, dataset, scene)
    # fname = 'r_125'
    # idx = '125'
    #
    # save_path = os.path.join(load_path, 'v15')
    #
    # u_file = os.path.join(load_path, 'activenerf','eval',f'uncertainty_{idx}_{fname}.npz')
    # u_dict = np.load(u_file,allow_pickle=True)
    # activenerf = u_dict['uncertanity_map']
    # activenerf = np.exp(activenerf)
    # rests = u_dict['rests'].tolist()
    # err = rests['rgb_err']
    #
    # mask = (err > 0.)
    # save_name = f'activegs_{fname}.jpg'
    # plot_unmap(save_path, torch.from_numpy(activenerf), mask, save_name, q=0.8)
    # #
    # save_name = f'activegs_{fname}_err.jpg'
    # plot_unmap(save_path, err, mask, save_name, q=0.8)
    # #
    # u_file = os.path.join(load_path, 'activevcurf3', 'eval', f'uncertainty_{idx}_{fname}.npz')
    # u_dict = np.load(u_file, allow_pickle=True)
    # vcurf = u_dict['uncertanity_map']
    # vcurf = np.exp(vcurf)
    # rests = u_dict['rests'].tolist()
    # err = rests['rgb_err']
    #
    # mask = (err > 0.)
    # save_name = f'vcurf_mlp_{fname}.jpg'
    # plot_unmap(save_path, torch.from_numpy(vcurf), mask, save_name, q=0.8)
    #
    # save_name = f'vcurf_mlp_{fname}_err.jpg'
    # plot_unmap(save_path, err, mask, save_name, q=0.8)

    # nerf

    output_path = "/mnt/Data2/liyan/MF-NeRF/results/colmap/"
    dataset = 'nerf_llff/NGP/fewshot15/'
    scene = 'room'
    load_path = os.path.join(output_path, dataset, scene)
    idx = '004'

    save_path = os.path.join(load_path, 'papers')

    e_file = os.path.join(load_path, f'{idx}_e.npy')
    err = np.load(e_file,allow_pickle=True)
    mask = (err>0.)
    save_name = os.path.join(save_path,f'e_{idx}.jpg')
    plot_unmap(save_name, torch.from_numpy(err), torch.from_numpy(mask), q=0.8)

    method = 'mcd_d'
    u_file = os.path.join(load_path, f'{idx}_{method}_u.npy')
    unc = np.load(u_file,allow_pickle=True)
    unc = unc.reshape(err.shape)
    save_name = os.path.join(save_path, f'{method}_{idx}.jpg')
    plot_unmap(save_name, torch.from_numpy(unc), torch.from_numpy(mask), q=0.8)

    method = 'mcd_r'
    u_file = os.path.join(load_path, f'{idx}_{method}_u.npy')
    unc = np.load(u_file, allow_pickle=True)
    unc = unc.reshape(err.shape)
    save_name = os.path.join(save_path, f'{method}_{idx}.jpg')
    plot_unmap(save_name, torch.from_numpy(unc), torch.from_numpy(mask), q=0.8)

    method = 'entropy'
    u_file = os.path.join(load_path, f'{idx}_{method}_u.npy')
    unc = np.load(u_file, allow_pickle=True)
    unc = unc.reshape(err.shape)
    save_name = os.path.join(save_path, f'{method}_{idx}.jpg')
    plot_unmap(save_name, torch.from_numpy(unc), torch.from_numpy(mask), q=0.7)

    method = 'warp'
    u_file = os.path.join(load_path, f'{idx}_{method}_u.npy')
    unc = np.load(u_file, allow_pickle=True)
    unc = unc.reshape(err.shape)
    save_name = os.path.join(save_path, f'vcurf_{idx}.jpg')
    plot_unmap(save_name, torch.from_numpy(unc), torch.from_numpy(mask), q=0.6)

    breakpoint()

    roc_file = os.path.join(load_path, f'{idx}_roc.npz')
    roc_dict = np.load(roc_file, allow_pickle=True)['roc_dict'].tolist()










