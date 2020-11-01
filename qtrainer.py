import os

import numpy as np
import torch

from tqdm import tqdm


class qTrainer(object):
    """
    Trainer需要承担两项职能。
    基础职能：训练与更新。
    额外职能：保存与恢复。
    使用前需要train或者eval。
    需要额外定义一个batch的处理步骤，继承此类，覆写step_one_batch函数即可。
    """

    def __init__(self, dataloader, model, criterion, optimizer, validation, device, n_epochs, save_dir, save_interval):
        # 训练相关
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.validation = validation

        self.n_epochs = n_epochs
        self.st_epoch = 0

        # IO相关
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_interval = save_interval

        #辅助
        self.training = True
        self.title = 'train'

    def classic_train(self, load_path=None):
        print('Manner : classic training\n')

        best_epoch = -1
        best_metric = -1e8

        if load_path:
            self.load_trainer(load_path)

        for epoch in range(self.st_epoch, self.n_epochs):
            self.epoch = epoch #记录当前轮次
            pbar = tqdm(self.dataloader)

            for i, data in enumerate(pbar):
                # trainer的问题在于，我们不知道数据传入的方式，有时候可能会需要input,mask,target.
                # 所以允许重载step_one_epoch函数来达到处理特殊格式数据的问题。
                batch_loss,metric_val = self.step_one_batch(i, data)

                if metric_val is not None:
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_epoch = epoch

                title = 'train' if self.training else 'infer'
                msg = "{0} epoch{1} loss:{2:.4f} ".format(title, epoch, batch_loss)
                if metric_val is not None:
                    msg += ' metric:{:.4f}'.format(metric_val)
                pbar.set_description(msg)

            if (epoch+1) % self.save_interval == 0:
                self.save_trainer()

        if best_epoch > -1:
            print('best epoch:{0}, best metric:{1}'.format(best_epoch,best_metric))

        return self

    def step_one_batch(self, i, data):
        """default func"""
        inpt, target = data  # 默认约定data格式为(input,target)
        pred = self.model(inpt)
        batch_loss = self.criterion(pred, target)

        self.optimizer.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

        metric_val = None
        return batch_loss.detach().cpu().numpy(), metric_val

    def save_trainer(self):
        """
        """
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'n_epochs': self.n_epochs
        }
        save_path = os.path.join(self.save_dir, '{0}.ckp'.format(self.epoch))
        print('Saving to {0} ...'.format(save_path))

        torch.save(state, save_path)
        print('Succeed!')

    def load_trainer(self, load_path):
        print('Loading from {0} ...'.format(load_path))
        checkpoint = torch.load(load_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.st_epoch = checkpoint['epoch'] + 1
        print('Succeed!')
        return self

    def train(self, mode=True):
        self.training = mode
        self.title = 'train' if self.training else 'infer'
        if mode:
            self.model.train()
        else:
            self.model.eval()
        return self

    def eval(self):
        self.train(mode=False)
        return self

    def __repr__(self):
        print(self.optimizer.__class__)

    def __call__(self, manner='classic', *args, **kwargs):

        if manner == 'classic':
            self.classic_train(*args, **kwargs)
        else:
            raise ValueError('unknown train manner')


class GCNTrainer(qTrainer):
    def __init__(self,*args,metric,idx_val,label_val,**kwargs):
        super(GCNTrainer,self).__init__(*args,**kwargs)
        self.metric = metric
        self.idx_val = idx_val
        self.label_val = label_val

    def step_one_batch(self, i, data):
        inpt, target, idx = data

        pred = self.model(inpt)
        batchloss = self.criterion(pred[idx], target[idx])

        self.optimizer.zero_grad()
        batchloss.backward()
        self.optimizer.step()

        """"
        metric
        """
        pred= pred.detach()
        acc = self.metric(pred[self.idx_val],self.label_val)
        #print(i,'acc: {:.4f}'.format(acc))

        return batchloss.detach().cpu().numpy(), acc
