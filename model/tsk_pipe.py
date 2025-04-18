import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from torch.optim import AdamW


from model.conponent.antecedent import (
    AntecedentSONFIN,
    antecedent_init_center_class_kmeans,
)
from model.callbacks import EarlyStoppingACC
from utils.util import NumpyDataLoader
from model.conponent.auto_tsk import AutoTSK
from model.trainer import Trainer


class TSKPipe:
    def __init__(
        self,
        n_rule=5,
        lr=0.01,
        weight_decay=1e-8,
        order=1,
        n_class=2,
        epochs=20,
        patience=20,
        verbose=1,
        alpha_threhold=0.5,
        device = 'cuda',
        logger = None
    ):
        self.n_rule = n_rule
        self.lr = lr
        self.weight_decay = weight_decay
        self.order = order
        self.n_class = n_class
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.wrapper = None
        self.alpha_threhold = alpha_threhold
        self.device = device
        self.logger = logger

    def initialize_centers(self,R, D):

        m = np.zeros((R)) 
        for r in range(1, R + 1):
            m[r - 1] = (r - 1) / (R - 1)
        return m

    def fit_autoTSK_search(
        self, X, y, gate_to_consequent=True, file_name=None, with_LN=True, l1 = 1e-8, l2 = 1e-7
    ):
        self.with_LN = with_LN
        print("AutoTSK search")
        # k-means init
        # init_center, init_sigma = antecedent_init_center_class_kmeans(X, y, class_num=self.n_rule)
        # random init
        # init_center = self.initialize_centers(self.n_rule, X.shape[1])

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,random_state=2025)

        if with_LN:
            gmf = nn.Sequential(
                AntecedentSONFIN(
                    in_dim=X_train.shape[1],
                    n_rule=self.n_rule,
                    # init_center=init_center,
                    # init_sigma=1.0,
                ),
                nn.LayerNorm(self.n_rule),
                nn.ReLU(),
            )
        else:
            gmf = nn.Sequential(
                AntecedentSONFIN(
                    in_dim=X_train.shape[1],
                    n_rule=self.n_rule,
                    # init_center=init_center,
                    # init_sigma=1.0,
                ),
            )
        print('gmf:',next(gmf.parameters()).device)

        # Define init sonfin model
        self.model = AutoTSK(
            in_dim=X_train.shape[1],
            out_dim=self.n_class,
            n_rule=self.n_rule,
            antecedent=gmf,
            order=self.order,
            gate_to_conseq=False,
            attention_to_conseq=False,
        )

        # Setup optimizer
        ante_param, gate_param, other_param = [], [], []
        aa= self.model.named_parameters()
        for name, param in self.model.named_parameters():
            if "center" in name or "sigma" in name:
                ante_param.append(param)
            elif 'alpha' in name:
                gate_param.append(param)
            else:
                other_param.append(param)

        optimizer1 = AdamW(
            [
                {"params": ante_param, "weight_decay": 0},
                {'params': gate_param, "weight_decay": 0},
                {"params": other_param, "weight_decay": 0},
            ],
            lr=self.lr,
        )
        early_stopping = EarlyStoppingACC(
            X_train, y_train, X_val, y_val, verbose=self.verbose, patience=self.patience)


        self.model.antecedent[0].set_using_alpha()


        tsk_trainer = Trainer(
            self.model,
            optimizer=optimizer1,
            criterion=nn.CrossEntropyLoss(),
            epochs=self.epochs,
            callbacks=[early_stopping],
            l1=l1,
            l2=l2,
            device="cuda",
            reset_param = True
        )  
        self.model = tsk_trainer.fit(X_train, y_train)


    def fit_autoTSK_retrain(
        self, X, y, gate_for_consequent=True, file_name=None, logger = None
    ):

        print("AutoTSK retrain")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1,random_state=2025)



        early_stopping = EarlyStoppingACC(
            X_train, y_train, X_val, y_val, verbose=self.verbose, patience=self.patience
        )


        with torch.no_grad():

            alpha_value_aftet_train = self.model.antecedent[0].alpha.data.clone()
            alpha_value_aftet_train[
                (alpha_value_aftet_train > self.alpha_threhold)
            ] = 1
            alpha_value_aftet_train[
                alpha_value_aftet_train <= self.alpha_threhold
            ] = 0
            
            self.model.antecedent[0].alpha_gate = alpha_value_aftet_train


            
            column_sums = alpha_value_aftet_train.sum(dim=0)
            remain_rule_num = (column_sums > 0).sum()
            print('remain_rule_num:',remain_rule_num)

            alpha_value_aftet_train = self.model.antecedent[0].alpha.data
            alpha_value_aftet_train[
                alpha_value_aftet_train <= self.alpha_threhold
            ] = 0
            alpha_value_aftet_train[
                (alpha_value_aftet_train <= 1) & (alpha_value_aftet_train > 0)
            ] = 1
            self.model.antecedent[0].alpha.data = alpha_value_aftet_train


        ante_param, gate_param, other_param = [], [], []
        for name, param in self.model.named_parameters():
            if "center" in name or "sigma" in name:
                ante_param.append(param)
            elif "alpha" in name:
                continue
            else:
                other_param.append(param)

        optimizer2 = AdamW(
            [
                {"params": ante_param, "weight_decay": 0},
                {"params": other_param, "weight_decay": self.weight_decay},
            ],
            lr=self.lr
        )
        # USING GATE IN ANTECEDENT AND CONSEQUENT
        self.model.set_gate_to_conseq()

        self.model.antecedent[0].set_using_alpha()

        self.model.n_rule = remain_rule_num
        
        tsk_trainer = Trainer(
            self.model,
            optimizer=optimizer2,
            criterion=nn.CrossEntropyLoss(),
            epochs=self.epochs,
            callbacks=[early_stopping],
            device="cuda",
            reset_param = False
        ) 

        self.model = tsk_trainer.fit(X_train, y_train)


    def predict(self, X):
        if hasattr(self, "model"):

            X = X.astype("float32")
            test_loader = DataLoader(
                NumpyDataLoader(X),
                batch_size=512,
                shuffle=False,
                drop_last=False,
            )
            y_preds = []
            for inputs in test_loader:
                y_pred = None

                self.model.eval()
                y_pred = self.model(inputs.to(self.device)).detach().cpu().numpy()

                y_preds.append(y_pred)
            all_preds = np.concatenate(y_preds, axis=0)
            final_probs = softmax(all_preds, axis=1)

            return final_probs

        else:
            raise Exception(
                "The model is not trained yet. Please call the 'fit' method first."
            )
    
    def save_model(self, path):
        torch.save(self.model, path)