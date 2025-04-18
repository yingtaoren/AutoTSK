from sklearn.metrics import accuracy_score
import numpy as np


class Callback:

    def on_batch_begin(self, wrapper):
        pass

    def on_batch_end(self, wrapper):
        pass

    def on_epoch_begin(self, wrapper):
        pass

    def on_epoch_end(self, wrapper):
        pass


class EarlyStoppingACC(Callback):

    def __init__(self, X_train,y_train,X, y, patience=1, verbose=0, save_path=None, save_best = True, file_name = None, logger = None):
        super(EarlyStoppingACC, self).__init__()
        self.X, self.y = X, y
        self.verbose = verbose
        self.patience = patience
        self.best_acc = 0
        self.cnt = 0
        self.logs = []
        self.save_path = save_path
        self.best_model = None
        self.X_train = X_train
        self.y_train = y_train
        self.file_name = file_name
        self.logger = logger
        if self.logger is not None:
            self.dataset = logger.config['dataset']

    def on_epoch_end(self, wrapper):
        cur_log = {}
        y_pred = wrapper.predict(self.X).argmax(axis=1)
        y_train_pre = wrapper.predict(self.X_train).argmax(axis=1)
        acc_train= accuracy_score(y_true=self.y_train, y_pred=y_train_pre)
        acc = accuracy_score(y_true=self.y, y_pred=y_pred)
        if self.logger is not None:
            a = 1
            self.logger.log({f'{self.dataset}/acc_train':acc_train,
                             f'{self.dataset}/acc_test' :acc, 
                             f'{self.dataset}/epoch' :wrapper.cur_epoch})

        if acc > self.best_acc:
            self.best_acc = acc
            self.cnt = 0
            # save best model
            wrapper.best_model = wrapper.model
            if self.save_path is not None:
                wrapper.save(self.save_path)
        else:
            self.cnt += 1
            if self.cnt > self.patience:
                wrapper.stop_training = True

                cur_log["epoch"] = wrapper.cur_epoch
                cur_log["acc"] = acc
                cur_log["best_acc"] = self.best_acc
                self.logs.append(cur_log)
                print("[Epoch {:5d}] EarlyStopping Callback trainACC:{:.4f}, ACC: {:.4f}, Best ACC: {:.4f}".format(cur_log["epoch"],acc_train, cur_log["acc"], cur_log["best_acc"]))


        cur_log["epoch"] = wrapper.cur_epoch
        cur_log["acc"] = acc
        cur_log["best_acc"] = self.best_acc
        self.logs.append(cur_log)
        if self.verbose > 0:
            print("[Epoch {:5d}] EarlyStopping Callback trainACC:{:.4f}, ACC: {:.4f}, Best ACC: {:.4f}".format(cur_log["epoch"],acc_train, cur_log["acc"], cur_log["best_acc"]))