import torch
import numpy as np
from scipy.special import softmax
from torch.utils.data import DataLoader
from torch.optim import AdamW


from utils.util import NumpyDataLoader


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        batch_size=512,
        epochs=1,
        callbacks=None,
        label_type="c",
        device="cpu",
        optimizer_grda=None,
        reset_param=False,
        loss=None,
        l1=0,
        l2=0,
        **kwargs
    ):
        self.model = model
        self.optimizer = optimizer
        self.optimizer_grda = optimizer_grda
        self.criterion = criterion
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)
        self.label_type = label_type
        self.best_model = model
        self.stop_training = False
        self.l1 = l1
        self.l2 = l2
        self.reg = True

        if callbacks is None:
            self.callbacks = []
        elif isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            raise ValueError("callback must be a Callback object")
        self.reset_param = reset_param
        if self.reset_param:
            self.model.reset_parameters()
        self.cur_batch = 0
        self.cur_epoch = 0
        self.grad_sum_epoch = {'center':0,'sigma':0,'alpha':0}

        self.kwargs = kwargs

    def train_on_batch(self, input, target):

        # update model one batch
        input, target = input.to(self.device), target.to(self.device)
        outputs = self.model(input)                    

        loss = (
            self.criterion(outputs, target)
            + self.model.antecedent[0].alpha.abs().sum() * self.l1
            + 0.5 * ((self.model.antecedent[0].alpha) ** 2).sum() * self.l2
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if "center" in name:
                    self.grad_sum_epoch['center'] += param.grad.abs().sum().item()
                elif 'sigma' in name:
                    self.grad_sum_epoch['sigma'] += param.grad.abs().sum().item()
                elif 'alpha' in name:
                    self.grad_sum_epoch['alpha'] += param.grad.abs().sum().item()

                        
    def fit(self, X, y):

        X = X.astype("float32")
        if self.label_type == "c":
            y = y.astype("int64")
        elif self.label_type == "r":
            y = y.astype("float32")
        else:
            raise ValueError('label_type can only be "c" or "r"!')

        self.fit_loader(X, y)
        return self.best_model

    def fit_loader(self, X, y):

        epoch_count = 0
        train_loader = self.create_dataloader(
            X,
            y,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.stop_training = False
        for e in range(self.epochs):

        
            
            self.cur_epoch = e
            self.__run_callbacks__("on_epoch_begin")

            for inputs in train_loader:
                self.__run_callbacks__("on_batch_begin")

                self.model.train()
                self.train_on_batch(inputs[0], inputs[1])
                self.cur_batch += 1

                self.__run_callbacks__("on_batch_end")

            self.__run_callbacks__("on_epoch_end")

            epoch_count = epoch_count + 1
            if self.stop_training:
                break
        return self

    def create_dataloader(
        self, X, y, batch_size, shuffle=True, num_workers=0, drop_last=False
    ):
        return DataLoader(
            NumpyDataLoader(X, y),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
        )

    def predict_proba(self, X, y=None):

        if self.label_type == "r":
            raise ValueError('predict_proba can only be used when label_type="c"')
        y_preds = self.predict(X)
        return softmax(y_preds, axis=1)

    def __run_callbacks__(self, func_name):
        for cb in self.callbacks:
            getattr(cb, func_name)(self)

    def update_para(self):

        ante_param, other_param = [], []
        for name, param in self.model.named_parameters():
            if "center" in name or "sigma" in name:
                ante_param.append(param)
            else:
                other_param.append(param)

        optimizer = AdamW(
            [
                {"params": ante_param, "weight_decay": 1e-4},
                {"params": other_param, "weight_decay": 1e-4},
            ],
            lr=0.01,
        )
        self.optimizer = optimizer
        return self

    def save(self, path):

        torch.save(self.model.state_dict(), path)

    def load(self, path):

        self.model.load_state_dict(torch.load(path))

    def predict(self, X, best=True):

        if best:
            X = X.astype("float32")
            test_loader = DataLoader(
                NumpyDataLoader(X),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.kwargs.get("num_workers", 0),
                drop_last=False,
            )
            y_preds = []
            for inputs in test_loader:
                y_pred = None
                if self.best_model is not None:
                    self.best_model.eval()
                    y_pred = (
                        self.best_model(inputs.to(self.device)).detach().cpu().numpy()
                    )

                else:
                    self.model.eval()
                    y_pred = self.model(inputs.to(self.device)).detach().cpu().numpy()

                y_preds.append(y_pred)
            all_preds = np.concatenate(y_preds, axis=0)
            final_probs = softmax(all_preds, axis=1)
            return final_probs 
