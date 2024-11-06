
import torch
import matplotlib.pyplot as plt

def compute_roc(opt,est,intervals = 10): # input torch.tensor
    ROC = []
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    thres = [torch.quantile(est, q / 100.0) for q in quants]
    subs = [est <= t for t in thres]
    ROC_points = [opt[s].mean().item() if s.any() else 0.0 for s in subs]
    ROC.extend(ROC_points)
    ROC_tensor = torch.tensor(ROC)
    AUC = torch.trapz(ROC_tensor, dx=1.0 / intervals).item()
    return ROC,AUC

def compute_ause(opt,est,intervals = 10): # input torch.tensor
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    thres = [torch.quantile(est, q / 100.0) for q in quants]

    opt_roc = []
    opt_subs = [opt <= t for t in thres]
    opt_points = [opt[s].mean().item() if s.any() else 0.0 for s in opt_subs]
    opt_roc.extend(opt_points)
    opt_roc = torch.tensor(opt_roc)

    est_roc = []
    est_subs = [est <= t for t in thres]
    est_points = [opt[s].mean().item() if s.any() else 0.0 for s in est_subs]
    est_roc.extend(est_points)
    est_roc = torch.tensor(est_roc)

    max_val = opt_roc.max()
    opt_roc_norm = opt_roc / max_val
    est_roc_norm = est_roc / max_val
    ause = torch.trapz(torch.abs(est_roc_norm-opt_roc_norm), dx=1.0 / intervals).item()

    return est_roc,ause

def plot_roc(ROC_dict,fig_name, opt_label='rgb_err',intervals = 10):
    quants = [100. / intervals * t for t in range(1, intervals + 1)]
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    # plot opt
    ROC_opt = ROC_dict.pop(opt_label)
    plt.plot(quants, ROC_opt, marker="^", markersize=10, linewidth= 2,color='blue', label=opt_label)
    for est_label in ROC_dict.keys():
        if 'rgb' in est_label:
            mark = "o"
        elif 'depth' in est_label:
            mark = "*"
        else:
            mark = "P"
        if 'l2' in est_label:
            line = '--'
        elif "var" in est_label:
            line = ':'
        else:
            line = '-.'
        plt.plot(quants, ROC_dict[est_label], marker=mark,linestyle=line, markersize=10,linewidth= 2, label=est_label)
    plt.xticks(quants)
    plt.xlabel('Sample Size(%)')
    plt.ylabel('Accumulative MSE')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(20, 12)
    fig.savefig(fig_name)
    plt.close()

def write_auc(AUC_dict,txt_name):
    with open(txt_name,'a') as f:
        for val in AUC_dict.keys():
            f.write(f'{val}: {AUC_dict[val]} \n')
        f.close()