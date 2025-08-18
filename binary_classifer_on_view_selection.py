import torch
from torch import nn
import os
# data
from torch.utils.data import TensorDataset,DataLoader

# pytorch-lightning
import pytorch_lightning
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import *
import time

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import argparse
import random

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
    moments = ['d_mean','d_var','d_skewness','d_kurtosis',
               'c_mean','c_var','c_skewness','c_kurtosis']

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


def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='csv file to create dataset')
    parser.add_argument('--dataset_name', type=str, default='m360',
                        choices=['m360', 'blender'],
                        help='which dataset to train/test')
    parser.add_argument('--scene', type=str, required=True,default='bicycle',
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
    def __init__(self, indim=3*10, n_classes=2,act='Sigmoid'):
        super().__init__()

        self.fc1 = nn.Linear(indim, 32)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(32, 64)  # Fully connected layer 2
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 8)
        self.fc6 = nn.Linear(8, n_classes)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x

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
        self.model = BinarryClassifier(indim=24, n_classes=2)

    def forward(self, features):
        return self.model(features)

    def setup(self, stage):

        self.train_dataset,self.test_dataset = create_dataset(data_file=self.hparams.data_file,
                                                              scene=self.hparams.scene,
                                                              target=self.hparams.target)


    def configure_optimizers(self):

        load_ckpt(self.model, self.hparams.ckpt_path)

        # opts = []
        self.net_opt = torch.optim.SGD(self.model.parameters(), self.hparams.lr)
        # opts += [self.net_opt]
        # net_sch = {
        #     'scheduler': torch.optim.lr_scheduler.StepLR(self.net_opt,1000,0.1),
        #     'interval': 'step',  # or 'epoch'
        #     'frequency': 1
        # }

        return self.net_opt

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

    def training_step(self, batch, batch_nb, *args):
        inputs,targets = batch
        logits = self(inputs)
        if isinstance(self.loss, nn.CrossEntropyLoss):
            loss = self.loss(logits, targets.long().to(logits.device))
            preds = logits.argmax(dim=1)
        elif isinstance(self.loss, nn.BCEWithLogitsLoss):
            # if you choose BCE path, change the model to n_classes=1 and adapt labels to shape [B, 1]
            probs = torch.sigmoid(logits.squeeze(-1))
            loss = self.loss(logits.squeeze(-1), targets.float().to(logits.device))
            preds = (probs >= 0.5).long()
        else:
            raise RuntimeError("Unsupported loss setup")

        acc = (preds == targets.to(preds.device)).float().mean().item()
        self.log('train/loss', float(loss))
        self.log('train/accuracy', acc, prog_bar=True)
        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        # self.val_dir = f'results/{self.hparams.exp_name}'
        # os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        torch.cuda.empty_cache()
        inputs, targets = batch
        logits = self(inputs)
        preds = logits.argmax(dim=1)
        correct = (preds == targets.to(preds.device)).sum()
        n_samples = torch.tensor(targets.size(0), device=logits.device)

        # for precision/recall/f1 in binary with class "1" as positive:
        tp = ((preds == 1) & (targets.to(preds.device) == 1)).sum()
        pos = (targets.to(preds.device) == 1).sum()

        return {'correct': correct, 'n_samples': n_samples, 'tp': tp, 'pos': pos}

    def on_validation_epoch_end(self, outputs):
        ## compute accuracy
        corrects = torch.stack([x['correct'] for x in outputs])
        n_samples = torch.stack([x['n_samples'] for x in outputs])
        tps = torch.stack([x['tp'] for x in outputs])
        pos = torch.stack([x['pos'] for x in outputs])

        total_correct = all_gather_ddp_if_available(corrects).sum()
        total_samples = all_gather_ddp_if_available(n_samples).sum()
        total_tp = all_gather_ddp_if_available(tps).sum()
        total_pos = all_gather_ddp_if_available(pos).sum()

        accuracy = total_correct / (total_samples + 1e-10)
        precision = total_tp / (total_tp + (total_pos - total_tp) + 1e-10)  # TP / (TP+FP)  (needs FP if you track it)
        recall = total_tp / (total_pos + 1e-10)  # TP / (TP+FN)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/f1', f1, prog_bar=True)

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

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    os.makedirs(os.path.join(f"logs/{hparams.dataset_name}", hparams.exp_name), exist_ok=True)
    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    else:
        strategy = "auto"  # or None (omit the arg)

    trainer = Trainer(max_epochs=0 if hparams.val_only else hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
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

