import torch
from torch import nn
import torch.nn.functional as F
import os
# data
from torch.utils.data import TensorDataset,DataLoader

# pytorch-lightning
import pytorch_lightning
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import wandb

from utils import *
import time

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import argparse
import random
import math


def sanitize_list(x):
    return [v if math.isfinite(v) else 0.0 for v in x]

def all_gather_ddp_if_available(x, cat_dim=0):
    """
    Replacement for the removed PL util.
    - If DDP initialized: all_gather across ranks and cat on `cat_dim`
    - Else: return x
    Requires `x` to be a Tensor with same shape on each rank.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        bufs = [torch.empty_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(bufs, x.contiguous())
        return torch.cat(bufs, dim=cat_dim)
    return x

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    checkpoint_ = {}
    if 'state_dict' in checkpoint: # if it's a pytorch-lightning checkpoint
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):
            continue
        k = k[len(model_name)+1:]
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):
                break
        else:
            checkpoint_[k] = v
    return checkpoint_


def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    if not ckpt_path: return
    model_dict = model.state_dict()
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)
    model_dict.update(checkpoint_)
    model.load_state_dict(model_dict)

def str2float(strlist):
    strlist = strlist[1:-1].split(',')
    return [float(x.strip()) for x in strlist]

def create_dataset(data_file,scene='bicycle',target='psnr',seed=0):
    # read in and create data
    raw_df = pd.read_csv(data_file)
    # scene_df = raw_df[raw_df['scene'] == scene]
    scene_df = raw_df
    features = ['d_mean','d_var',
               'c_mean','c_var','nv_pxs'] + [f'c_hist_{i}' for i in range(10)] + [f'd_hist_{i}' for i in range(10)]


    # dataset
    x_data = []
    y_data = []
    n_views = len(scene_df)
    for i in range(n_views):
        for j in range(n_views):
            if not i==j:
                f1 = [float(scene_df[key].iloc[i]) for key in moments]
                f2 = [float(scene_df[key].iloc[j]) for key in moments]
                diff = [a - b for a, b in zip(f1, f2)]
                psnr1 = float(scene_df[target].iloc[i])
                psnr2 = float(scene_df[target].iloc[j])
                label = int(psnr1<psnr2)

                features = f1 + f2 + diff
                x_data.append(features)
                y_data.append(label)

    # after your loop
    data = list(zip(x_data, y_data))  # pair them together

    # shuffle in-place
    random.seed(seed)
    random.shuffle(data)

    # split back
    x_data, y_data = zip(*data)

    # now take every 8th as test, rest as train
    x_train, y_train, x_test, y_test = [], [], [], []

    for idx, (x, y) in enumerate(zip(x_data, y_data)):
        if idx % 8 == 0:  # every 8th sample → test
            x_test.append(x)
            y_test.append(y)
        else:  # rest → train
            x_train.append(x)
            y_train.append(y)

    x_train = torch.tensor(x_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset,test_dataset

def create_dataset_multi_scenes(data_dir,timesteps,scenes,target='psnr',seed=0):
    features = ['d_mean', 'd_var',
                'c_mean', 'c_var', 'nv_pxs'] + [f'c_hist_{i}' for i in range(10)] + [f'd_hist_{i}' for i in range(10)]
    x_data = []
    y_data = []

    for timestep in timesteps:
        for scene in scenes:
            curr_data_file = f'{data_dir}_{timestep}/{scene}/candidates.csv'
            curr_df = pd.read_csv(curr_data_file)
            n_views = len(curr_df)
            for i in range(n_views):
                for j in range(n_views):
                    if not i==j:
                        f1 = [float(curr_df[key].iloc[i]) for key in features]
                        f2 = [float(curr_df[key].iloc[j]) for key in features]
                        diff = [a - b for a, b in zip(f1, f2)]
                        psnr1 = float(curr_df[target].iloc[i])
                        psnr2 = float(curr_df[target].iloc[j])

                        f_cat = f1 + f2 + diff
                        label = int(psnr1<psnr2)

                        x_data.append(f_cat)
                        y_data.append(label)

    # after your loop
    data = list(zip(x_data, y_data))  # pair them together

    # shuffle in-place
    random.seed(seed)
    random.shuffle(data)

    # split index (80% train, 20% test)
    split_index = int(len(data) * 0.8)

    train_data = data[:split_index]
    test_data = data[split_index:]

    # unzip back into x and y
    x_train, y_train = zip(*train_data)
    x_test, y_test = zip(*test_data)

    x_train = torch.tensor(x_train,dtype=torch.float32)
    x_train = torch.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = torch.tensor(y_train,dtype=torch.long)
    train_dataset = TensorDataset(x_train, y_train)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    x_test = torch.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(x_test, y_test)

    return train_dataset,test_dataset


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory to save csv file to create dataset')
    parser.add_argument('--data_timesteps',nargs="+", type=str, required=True,
                        help='timestep to save csv file to create dataset')
    parser.add_argument('--dataset_name', type=str, default='m360',
                        choices=['m360', 'blender'],
                        help='which dataset to train/test')
    parser.add_argument('--scenes', nargs="+", type=str, required=True,default=['bicycle','kitchen'],
                        choices=['kitchen','garden','bicycle','counter','bonsai','flowers','room','stump','all'],
                        help='run on which scene')
    parser.add_argument('--target', type=str, default='psnr',
                        choices=['psnr','ssim','lpips'],
                        help='train the classifier based on which metric')

    # training options
    parser.add_argument('--batch_size', type=int, default=128,
                        help='number of samples in a batch')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')

    # loss options
    parser.add_argument('--loss', type=str, default='ce',
                        choices=['bce', 'nll','ce'],
                        help='which loss to train')

    # validation options
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument("--seed", type=int, default=0,
                        help='random seed to initialize the training set')

    return parser.parse_args()

class BinarryClassifier(nn.Module):
    def __init__(self, indim=25*3, n_classes=2, act=None, use_input_bn=True):
        super().__init__()

        # optional final activation (usually leave None when using CrossEntropyLoss)
        if act == 'Sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'Softmax':
            self.act = nn.Softmax(dim=-1)
        elif act == 'logSoftmax':
            self.act = nn.LogSoftmax(dim=-1)
        else:
            self.act = None

        self.use_input_bn = use_input_bn
        if self.use_input_bn:
            self.bn_in = nn.BatchNorm1d(indim, eps=1e-5, momentum=0.1)

        self.fc1 = nn.Linear(indim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.fc6 = nn.Linear(16, n_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # (optional) init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_input_bn:
            x = self.bn_in(x)

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.relu(self.bn5(self.fc5(x)))
        x = self.dropout(x)
        x = self.fc6(x)  # (B, 2) for CE

        if self.act is not None:
            x = self.act(x)  # only if you explicitly want it (e.g., BCE or eval-time probs)
        return x



# class ResidualBlock(nn.Module):
#     def __init__(self, dim, dropout=0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.SiLU(),
#             nn.Linear(dim, dim),
#             nn.SiLU(),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         return x + self.net(x)  # residual

# class BinaryClassifier(nn.Module):
#     def __init__(self, in_dim=25*3, width=128, depth=3, dropout=0.2):
#         """
#         in_dim:   input feature dimension
#         width:    hidden width
#         depth:    number of residual blocks
#         dropout:  dropout prob
#         """
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Linear(in_dim, width),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(depth)])
#         self.head = nn.Sequential(
#             nn.LayerNorm(width),
#             nn.ReLU(),
#             nn.Linear(width, width // 2),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(width // 2, 2)  # single logit for binary
#         )
#
#         # Kaiming init for linear layers
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
#
#     def forward(self, x):
#         x = self.stem(x)
#         x = self.blocks(x)
#         logit = self.head(x).squeeze(-1)  # shape: (B,)
#         return logit  # raw logits

class ViewClassifySystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 20
        self.update_interval = 16

        self.star_time = time.time()

        if self.hparams.loss == 'bce':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.hparams.loss == 'nll':
            self.loss = nn.NLLLoss()  # would require LogSoftmax in forward
        else:
            self.loss = nn.CrossEntropyLoss()
        self.model = BinarryClassifier(indim=25*3, n_classes=2)
        self._val_outputs = []

    def forward(self, features):
        return self.model(features)

    def setup(self, stage):

        self.train_dataset,self.test_dataset = create_dataset_multi_scenes(
                                                              data_dir=self.hparams.data_dir,
                                                              timesteps=self.hparams.data_timesteps,
                                                              scenes=self.hparams.scenes,
                                                              target=self.hparams.target)

    def configure_optimizers(self):
        # 1) optional checkpoint load
        ckpt_path = getattr(self.hparams, "ckpt_path", None)
        if ckpt_path:
            load_ckpt(self.model, ckpt_path)

        # 2) optimizer hyperparams (with safe defaults)
        lr = float(getattr(self.hparams, "lr", 3e-4))
        wd = float(getattr(self.hparams, "weight_decay", 1e-2))
        betas = getattr(self.hparams, "betas", (0.9, 0.999))
        eps = float(getattr(self.hparams, "adam_eps", 1e-8))

        # 3) param groups: no weight decay on bias/LayerNorm
        no_decay_keys = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight")
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if any(k in n for k in no_decay_keys) else decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": wd},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=lr, betas=betas, eps=eps,
        )

        # 4) scheduler: prefer OneCycle (step-wise) if steps_per_epoch is known
        max_epochs = getattr(self.trainer, "max_epochs", None) if hasattr(self, "trainer") else None
        steps_per_epoch = None
        if hasattr(self, "trainer") and self.trainer is not None and max_epochs:
            # Lightning sets this; safe fallback if unavailable
            est_steps = getattr(self.trainer, "estimated_stepping_batches", None)
            if est_steps:
                steps_per_epoch = max(1, est_steps // max_epochs)

        if steps_per_epoch:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=max_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=1e2,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # update every step
                    "frequency": 1,
                },
            }

        # fallback: cosine warm restarts (epoch-wise)
        t0 = int(getattr(self.hparams, "cosine_t0", 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=t0)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=4,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=4,
                          batch_size=8,
                          pin_memory=True)

    def _get_ce_weight(self, device):
        w = getattr(self.loss, "weight", None)
        if w is None:
            return None
        w = w.to(device).float()
        # Replace NaN/Inf and cap extremes (helps when a class has 0 count)
        w = torch.nan_to_num(w, nan=1.0, posinf=1e3, neginf=0.0)
        w = w.clamp(min=0.0, max=1e3)
        return w

    def _compute_loss_and_preds(self, logits, targets):
        # --- Expect CrossEntropyLoss with TWO logits ---
        if not isinstance(self.loss, nn.CrossEntropyLoss):
            raise RuntimeError("This build expects CrossEntropyLoss with 2-logit outputs (B,2).")
        if logits.dim() != 2 or logits.size(-1) != 2:
            raise RuntimeError(f"Expected logits of shape (B,2); got {tuple(logits.shape)}")

        device = logits.device

        # --- normalize targets to (B,) long in {0,1} ---
        t = targets.to(device)
        if t.dim() == 2 and t.size(-1) == 2:  # one-hot -> indices
            t = t.argmax(dim=1)
        elif t.dim() == 2 and t.size(-1) == 1:  # (B,1) -> (B,)
            t = t.squeeze(-1)
        elif t.dim() > 1:  # fallback: flatten
            t = t.view(-1)
        t = t.long()

        # --- safe class weights ---
        weight = self._get_ce_weight(device)

        # --- compute per-sample CE to detect bad rows ---
        loss_per = F.cross_entropy(
            logits, t, weight=weight,
            reduction="none",
            label_smoothing=getattr(self.loss, "label_smoothing", 0.0)
        )

        finite_mask = torch.isfinite(loss_per)
        if not finite_mask.all():
            bad = int((~finite_mask).sum().item())
            self.log("debug/ce_bad_samples", bad, on_step=True, prog_bar=True)
            # keep only good samples
            loss_per = loss_per[finite_mask]
            t = t[finite_mask]
            logits = logits[finite_mask]

        if loss_per.numel() == 0:
            raise RuntimeError("All samples in this batch produced non-finite CE loss. "
                               "Check class weights and targets.")

        loss = loss_per.mean()

        # --- predictions & probabilities ---
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(class=1)
        preds = logits.argmax(dim=1)  # (B,)

        return loss, preds, probs

    def _sanitize_batch(self, inputs, targets, max_abs=1e6):
        # Replace NaN/Inf and clamp extremes to prevent fp16 overflow
        inputs = torch.nan_to_num(inputs, nan=0.0, posinf=max_abs, neginf=-max_abs)
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)

        # Remove any rows still containing non-finite values
        mask_x = torch.isfinite(inputs).all(dim=1)
        mask_y = torch.isfinite(targets.view(-1))
        mask = mask_x & mask_y
        if not mask.all():
            inputs, targets = inputs[mask], targets[mask]

        # Optional safety clamp
        inputs = inputs.clamp_(-max_abs, max_abs)
        return inputs, targets

    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch

        # 1) sanitize
        inputs, targets = self._sanitize_batch(inputs, targets)

        # 2) forward
        logits = self.model(inputs)

        # 3) compute loss/preds (your helper already handles CE vs BCE)
        loss, preds, probs = self._compute_loss_and_preds(logits, targets)

        # 4) catch non-finite loss early (dump stats to help debug)
        if not torch.isfinite(loss):
            with torch.no_grad():
                self.log("debug/nonfinite_loss", 1, on_step=True, prog_bar=True)

                # sanitize to avoid NaN-aware ops not available in older torch
                x_s = torch.nan_to_num(inputs, nan=0.0, posinf=0.0, neginf=0.0)
                l_s = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

                # stats without nanstd/nanmean
                x_mean = x_s.mean().item()
                x_std = x_s.std(unbiased=False).item()
                x_min = x_s.amin().item()
                x_max = x_s.amax().item()

                logit_mean = l_s.mean().item()
                logit_std = l_s.std(unbiased=False).item()
                logit_min = l_s.amin().item()
                logit_max = l_s.amax().item()

            raise RuntimeError(
                f"Non-finite loss. "
                f"x(mean={x_mean:.3e}, std={x_std:.3e}, min={x_min:.3e}, max={x_max:.3e}) | "
                f"logits(mean={logit_mean:.3e}, std={logit_std:.3e}, min={logit_min:.3e}, max={logit_max:.3e})"
            )

        # 5) metrics
        acc = (preds == targets.to(preds.device).view(-1).long()).float().mean()
        if hasattr(self, "net_opt"):
            self.log("train/lr", self.net_opt.param_groups[0]["lr"], on_step=True, prog_bar=False)

        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/accuracy_step", acc, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/accuracy_epoch", acc, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        # self.val_dir = f'results/{self.hparams.exp_name}'
        # os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self.model(inputs)  # keep consistent with training
        loss, preds, probs = self._compute_loss_and_preds(logits, targets)

        t_long = targets.to(preds.device).view(-1).long()
        correct = (preds == t_long).sum()
        n = torch.tensor(t_long.numel(), device=preds.device)

        # binary stats (take class "1" as positive)
        tp = ((preds == 1) & (t_long == 1)).sum()
        fp = ((preds == 1) & (t_long == 0)).sum()
        fn = ((preds == 0) & (t_long == 1)).sum()
        pos = (t_long == 1).sum()

        # Log per-step loss for monitoring
        self.log("val/loss_step", loss, prog_bar=False, on_step=True, on_epoch=False)

        out = {
            "val_loss": loss.detach(),
            "correct": correct.detach(),
            "n": n.detach(),
            "tp": tp.detach(),
            "fp": fp.detach(),
            "fn": fn.detach(),
            "pos": pos.detach(),
        }
        self._val_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        outputs = self._val_outputs
        if not outputs:
            return

        device = outputs[0]["n"].device

        # accumulate sums
        total_n = torch.stack([o["n"].to(device) for o in outputs]).sum()
        total_correct = torch.stack([o["correct"].to(device) for o in outputs]).sum()
        total_tp = torch.stack([o["tp"].to(device) for o in outputs]).sum()
        total_fp = torch.stack([o["fp"].to(device) for o in outputs]).sum()
        total_fn = torch.stack([o["fn"].to(device) for o in outputs]).sum()
        total_pos = torch.stack([o["pos"].to(device) for o in outputs]).sum()

        # sample-weighted loss average
        # assumes each "val_loss" is mean over the batch; weight by batch size
        batch_losses = torch.stack([o["val_loss"].to(device) for o in outputs])
        batch_ns = torch.stack([o["n"].to(device).float() for o in outputs])
        val_loss = (batch_losses * batch_ns).sum() / batch_ns.clamp_min(1).sum()

        # metrics
        acc = total_correct.float() / total_n.clamp_min(1)
        precision = total_tp.float() / (total_tp + total_fp).clamp_min(1)
        recall = total_tp.float() / total_pos.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)

        # epoch-level logging
        self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/accuracy", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/precision", precision, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/recall", recall, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/f1", f1, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # optional: clear cache & (if you really must) free memory now
        self._val_outputs.clear()

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    start = time.time()
    hparams = get_opts()

    pytorch_lightning.seed_everything(hparams.seed)
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')

    if hparams.val_only:
        system = ViewClassifySystem.load_from_checkpoint(hparams.ckpt_path, strict=False, hparams=hparams)
    else:
        system = ViewClassifySystem(hparams)

    ckpt_path = f'ckpts/{hparams.dataset_name}/{hparams.exp_name}'
    os.makedirs(ckpt_path, exist_ok=True)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    os.makedirs(os.path.join(f"logs/{hparams.dataset_name}", hparams.exp_name), exist_ok=True)
    tb_logger = TensorBoardLogger(
        save_dir=f"logs/{hparams.dataset_name}",
        name=hparams.exp_name,
        default_hp_metric=False
    )

    wandb_logger = WandbLogger(
        project=hparams.dataset_name,  # or a custom project name
        name=hparams.exp_name
    )

    wandb_logger.experiment.config.update(vars(hparams), allow_val_change=True)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"  # or None (omit the arg)

    trainer = Trainer(max_epochs=0 if hparams.val_only else hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=[tb_logger, wandb_logger],
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=strategy,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision="16-mixed")

    trainer.fit(system)

    end = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(end - start))
    print('Total runtime: {}'.format(runtime))

