import torch
from torch import nn
# data
from torch.utils.data import TensorDataset,DataLoader

# pytorch-lightning
import pytorch_lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import *
import time

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import argparse
import random

def str2float(strlist):
    strlist = strlist[1:-1].split(',')
    return [float(x.strip()) for x in strlist]
def create_dataset(data_file,scene='bicycle',target='psnr',seed=0):
    # read in and create data
    raw_df = pd.read_csv(data_file)
    scene_df = raw_df[raw_df['scene'] == scene]
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
                if psnr1>psnr2:
                    label = [1,0]
                else:
                    label = [0,1]

                x_data.append(diff)
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
    y_train = torch.tensor(y_train,dtype=torch.float32)
    train_dataset = TensorDataset(x_train, y_train)

    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
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
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'nll'],
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
        if act=='Sigmoid':
            self.act = nn.Sigmoid()
        if act=='Softmax':
            self.act = nn.Softmax()
        if act == 'logSoftmax':
            self.act = nn.LogSoftmax()

        self.fc1 = nn.Linear(indim, 16)  # Fully connected layer 1
        self.relu = nn.ReLU()  # ReLU activation
        self.fc2 = nn.Linear(16, 8)  # Fully connected layer 2
        self.fc3 = nn.Linear(8, n_classes)
        self.dropout = nn.Dropout(p=0.2)
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.act(x)
        return x

class ViewClassifySystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 20
        self.update_interval = 16

        self.star_time = time.time()

        if self.hparams.loss == 'bce':
            self.loss = nn.BCELoss()
        if self.hparams.loss == 'nll':
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.model = BinarryClassifier(indim=10, n_classes=1,act='Softmax')

    def forward(self, features):
        return self.model(features)

    def setup(self, stage):

        self.train_dataset,self.test_dataset = create_dataset(data_file=self.hparams.data_file,
                                                              test_scene=self.hparams.test_scene,
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
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=8,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, *args):
        inputs,targets = batch
        inputs.requires_grad=True
        results = self(inputs)
        # targets = targets.type(torch.LongTensor)
        loss = self.loss(results, targets.to(results))
        print(loss)

        predicted_labels = (results >= 0.5).float()
        correct = (predicted_labels == targets).sum().item()
        total_samples = targets.size(0)

        accuracy = correct / total_samples
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss.item())
        self.log('train/accuracy', accuracy, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        # self.val_dir = f'results/{self.hparams.exp_name}'
        # os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        torch.cuda.empty_cache()
        inputs, targets = batch
        results = self(inputs)

        logs = {}

        predicted_labels = (results >= 0.5).float()
        correct = (predicted_labels == targets).sum()
        total_samples = targets.size(0)

        true_positives = ((predicted_labels == 1) & (targets == 1)).sum()
        total_positives = (targets == 1).sum()

        logs['correct'] = correct
        logs['n_samples'] = torch.tensor(total_samples)
        logs['true_positives'] = true_positives
        logs['total_positives'] = total_positives

        return logs

    def validation_epoch_end(self, outputs):
        ## compute accuracy
        corrects = torch.stack([x['correct'] for x in outputs])
        total_corrects = all_gather_ddp_if_available(corrects).sum()

        n_samples = torch.stack([x['n_samples'] for x in outputs])
        total_samples = all_gather_ddp_if_available(n_samples).sum()

        accuracy = total_corrects/total_samples
        self.log('test/accuracy', accuracy, True)

        true_positives = torch.stack([x['true_positives'] for x in outputs])
        total_true_positives = all_gather_ddp_if_available(true_positives).sum()

        total_positives = torch.stack([x['total_positives'] for x in outputs])
        total_total_positives = all_gather_ddp_if_available(total_positives).sum()

        precision = total_true_positives / (total_total_positives + 1e-10)  # to avoid division by zero
        recall = total_true_positives / (total_samples + 1e-10)  # to avoid division by zero
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)  #

        self.log('test/f1', f1, True)

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

    trainer = Trainer(max_epochs=0 if hparams.val_only else hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system)

    end = time.time()
    runtime = time.strftime("%H:%M:%S", time.gmtime(end - start))
    print('Total runtime: {}'.format(runtime))

