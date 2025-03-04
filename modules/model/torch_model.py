from datetime import datetime as dt
from typing import Callable, Union
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from distutils.dir_util import mkpath
from modules.model.algos.layers import BerDrop
from modules.model.model_wrapper import ModelWrapper
from modules.data_loading import generate_dataloader
from modules.transforms import fill_na
from modules.log import logger
from sklearn.metrics import r2_score as r2


class LightningModel(ModelWrapper):
    """Lightning Model Wrapper

    This is a Model Wrapper for a PyTorch Lighning model.

    It takes a model class and its arguments as parameters.

    The first parameters are always the feature and target columns of the data,
    validation and training indices and the actual scoring function for validation.

    Also max epochs and batch size for training is adjustable.

    The interesting part is the built-in early stopping, which 
    has a burn-in phase.
    An integer indicates the phase is in terations and a 
    floating point that it declared as fration of the training data.

    And the learning rate scheduler, which takes a class an its 
    parameters as input.
    Such scheduler is important to help the model not get stuck or
    overfits during training, it makes use of validation metric(s).
    
    
    """
    def __init__(
        self,
        feature_columns,
        target_columns,
        model_class,
        model_args: dict = None,  # type: ignore
        train_idx: list = None,  # type: ignore
        valid_idx: list = None,  # type: ignore
        score_fun: Callable = r2,
        multi_fit: bool = False,
        max_epochs=200,
        batch_size=256,
        early_stopping: bool = True,
        burn_in_steps: Union[int, float] = .25,
        scheduler: type = None,  # type: ignore
        scheduler_params: dict = dict(
            factor=0.75,
            patience=3,
            verbose=1,
            mode='min',
            cooldown=10,
            min_lr=1e-7
        ),
        callbacks: list = None,  # type: ignore
        constraint_x_cols: list = None  # type: ignore
    ):
        """

        Args:
            feature_columns (Iterable[str|int]): input feature indices or column names.
            target_columns (Iterable[str|int]): output value indices or column names.
            model_class (nn.Module): class of the underlying NN module.
            model_args (dict, optional): Initialization parameters of the underlying 
                NN module. Defaults to None.
            train_idx (list, optional): training datapoint indices. 
                Defaults to teh first 80%.
            valid_idx (list, optional): training datapoint indices. 
                Defaults to the last 20%.
            score_fun (Callable, optional): a scoring function for the validation step. 
                Defaults to r2.
            multi_fit (bool, optional): whether to use multi-fit or not (unsued here). 
                Defaults to False.
            max_epochs (int, optional): max training steps. Defaults to 200.
            batch_size (int, optional): batch size for training. Defaults to 256.
            early_stopping (bool, optional): whether to use early stopping. 
                Defaults to True.
            burn_in_steps (Union[int, float], optional): An integer indicates the phase 
                is in terations and a floating point that it declared as fration of the 
                training data. Defaults to .25.
            scheduler (type, optional): LR scheduler class. Defaults to None.
            scheduler_params (dict, optional): LR scheduler initialization parameters. 
                Defaults to 
                `dict(factor=0.75, patience=3, verbose=1, mode='min', cooldown=10, min_lr=1e-7)`.
            callbacks (list, optional): additional callbacks for training. Defaults to None.
            constraint_x_cols (list, optional): if attaching a constraint model, which columns are
                part of it to further constrain the model output during training. 
                Defaults to None.
        """
        super().__init__(feature_columns, target_columns)

        # hierarchy checked
        if 'scheduler_params' in model_args:
            if scheduler_params is None:
                scheduler_params = model_args['scheduler_params']
            model_args['scheduler_params'] = None
        if scheduler is None:
            scheduler = model_args.get('scheduler', ReduceLROnPlateau)

        # burn-in phase
        if burn_in_steps < 1:
            burn_in_steps = burn_in_steps * max_epochs

        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.score_fun = score_fun
        self.multi_fit = multi_fit
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.burn_in_steps = burn_in_steps
        self.scheduler_params = scheduler_params
        self.scheduler = scheduler
        self.callbacks = callbacks
        self.constraint_x_cols = constraint_x_cols
        self.trainer_a = None
        self.trainer_b = None
        self.epoch = 0

    def fit(self, data_train, data_valid=None):
        """
        Fit my lightning model.

        This does except two values, if given.
        The second is the path or data for the validation set.
        Per default it is `None`.

        If it is not set and we have a path to the train data, it's discarded.
        Else this models train_idx and valid_idx are used.

        If paths are set, paths to folders containing csv-files is expected.
        They are handled over to a `CSVDataSet`.

        If we have one or two DataFrames already, they are handled as `np.ndarray`
        into `TensorDataset`.

        :param data_train: Training data, either as DataFrame or path to a csv folder.
        :param data_valid: Validation data, either as DataFrame or path to a csv folder.
        :return:
        """
        # defaults
        x1 = data_train
        x2 = None
        # parsing
        if data_valid is None and self.train_idx is not None:
            if isinstance(data_train, pd.DataFrame):
                if self.valid_idx is None:
                    self.valid_idx = set(
                        data_train.index
                    ).difference(self.train_idx)
                x1 = data_train.loc[self.train_idx]
                x2 = data_train.loc[self.valid_idx]  # type: ignore
            elif isinstance(data_train, list):
                if self.valid_idx is None:
                    self.valid_idx = set(
                        list(range(len(data_train)))
                    ).difference(self.train_idx)
                x1 = data_train[self.train_idx]  # type: ignore
                x2 = data_train[self.valid_idx]  # type: ignore
        else:
            x1 = data_train
            x2 = data_valid

        # get data loaders
        tr_dl = generate_dataloader(
            x1, xcols=self.features, ycols=self.targets, batch_size=self.batch_size
        )
        if x2 is not None:
            va_dl = generate_dataloader(
                x2,
                xcols=self.features,
                ycols=self.targets,
                batch_size=self.batch_size,
                shuffle=False
            )
        else:
            va_dl = None

        # parsing missing validation set
        if va_dl is None:
            self.model_args['validation_string'] = 'train_loss'

        # constraint loss check & parsing
        if self.model_args.get('constraint_loss_model') is not None:
            if self.constraint_x_cols is not None:
                self.model_args['constraint_x_idx'] = np.where(list(map(
                    lambda f: f in self.constraint_x_cols,
                    self.features
                )))[0]

        self.model_args['scheduler_params'] = self.scheduler_params
        self.model_args['scheduler'] = self.scheduler
        if self.model is None:
            self.model = self.model_class(**self.model_args)

        callbacks = self.callbacks
        if self.early_stopping:
            early_stopping = EarlyStopping(
                getattr(self.model, 'validation_string', 'val_loss'),
                patience=self.scheduler_params.get('patience', 3) + 2,
                min_delta=self.scheduler_params.get('min_lr', 1e-8)
            )

        chpth = f'./checkpoints/{dt.now().timestamp()}/'
        mkpath(chpth)
        burnin = int(max(0, self.burn_in_steps))
        epochs = self.max_epochs - burnin

        if burnin > 0:
            self.trainer_b = pl.Trainer(
                max_epochs=burnin,
                log_every_n_steps=10,
                callbacks=callbacks
            )
            self.trainer_b.fit(
                self.model, tr_dl, va_dl,  # ckpt_path=chpth+'/net.cp'
            )
            self.lr_con = self.model.optimizers().param_groups[-1]['lr']
            logger.info(f'after burn-in lr {self.lr_con}')
            self.epoch += self.trainer_b.current_epoch
        else:
            self.trainer_b = None
            self.lr_con = self.model.learning_rate

        if self.early_stopping:
            if callbacks is None:
                callbacks = [early_stopping]
            else:
                callbacks += [early_stopping]

        self.trainer_a = pl.Trainer(
            max_epochs=epochs,
            log_every_n_steps=10,
            callbacks=callbacks  # type: ignore
        )
        self.trainer_a.fit(
            self.model, tr_dl, va_dl,  # ckpt_path=chpth+'/net.cp'
        )
        self.lr_con = self.model.optimizers().param_groups[-1]['lr']
        logger.info(f'final lr {self.lr_con}')
        self.epoch += self.trainer_a.current_epoch
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self. _check_features_(df)
        return pd.DataFrame(
            fill_na(self.model(
                torch.tensor(df[self.features].astype(float).fillna(0).values).float()
            ).detach().numpy()),  # type: ignore
            columns=self.targets
        )

    def score(self, df: pd.DataFrame):
        with torch.no_grad():
            return self.score_fun(
                fill_na(self.model(torch.tensor(
                    df[self.features].astype(float).fillna(0).values
                ).float()).detach().numpy()),  # type: ignore
                df[self.targets].astype(float).fillna(0).values
            )


# defining a simple Feed-Foreward Neural Network
class FFNN(ModelWrapper):
    """
    An example for a Feed-Forward Neural Net.

    The network has `len(n_hidden)` hidden layers
    with sizes defined by `n_hidden`.

    Each consecutive (h > 1) hidden layer starts with
    a 1D batch norm, followed by a linear layer.

    The final hidden or the input layer ends with a Sigmoid.
    And the last layer ends with a ReLU, due to the fact
    that we model only positive data at this point!

    Beneath the usual forward call, it has a fit function. (`self.fit`)
    This function does fit the model iteratively on some input data.
    It uses `lr_init` on a `Adagrad` optimizer.
    The loss function there is based

    :parameters:
        n_input: int, How many input/feature variables do we have.
        n_output: int, How many output/feature variables do we have.
        n_hidden: list, List of hidden layer sizes.
        lr_init: float, Initial learning rate for backprop.

    :methods:
    """
    def __init__(
        self,
        features,
        targets,
        n_hidden: list = [80, 40],
        lr_init: float = 1e-4,
        validation_metrics: dict = {
            'r2': lambda y, x: 1 - ((x-y)**2).sum() / y.var()
        },
        drop_flag: bool = True,
        drop_prob: float = 0.8,
        bernoulli_drop: bool = True
    ):
        n_input = len(features)
        n_output = len(targets)
        super().__init__(features, targets)
        self.lr_init = lr_init
        if isinstance(n_hidden, int):
            n_hidden = [n_hidden]
        if n_hidden is None:
            n_hidden = []

        self.model = nn.Sequential()
        self.model.append(nn.Tanh())

        if len(n_hidden) > 0:
            self.model.append(nn.Linear(n_input, n_hidden[0]))
        else:
            n_hidden = [n_input]

        if len(n_hidden) > 1:
            for h in range(1, len(n_hidden)):
                self.model.append(nn.BatchNorm1d(n_hidden[h-1]))
                self.model.append(nn.Linear(n_hidden[h-1], n_hidden[h]))
                if drop_flag:
                    if drop_prob is None:
                        drop_prob = .8
                    if bernoulli_drop:
                        self.model.append(BerDrop(1 - drop_prob))
                    else:
                        self.model.append(nn.Dropout(drop_prob))

        self.model.append(nn.Sigmoid())
        self.model.append(nn.Linear(n_hidden[-1], n_output))
        self.model.append(nn.ReLU())
        self.va_metrics = validation_metrics
        self.n_hidden = n_hidden
        self.lr_init = lr_init
        self.drop_flag = drop_flag
        self.drop_prob = drop_prob
        self.bernoulli_drop = bernoulli_drop

    def predict(self, data):
        data = self. _check_features_(data)
        return self.model(data[self.features])

    def get_params(self, **kwargs):
        return {
            "features": self.features,
            "targets": self.targets,
            "n_hidden": self.n_hidden,
            "lr_init": self.lr_init,
            "validation_metrics": self.va_metrics,
            "drop_flag": self.drop_flag,
            "drop_prob": self.drop_prob,
            "bernoulli_drop": self.bernoulli_drop
        }

    def fit(
        self,
        data,
        va_idx,
        n_iter: int = 1000,
        batch_size: int = 250,
        verbose: int = 50,
        gamma_max: float = 2.1,
        gamma_min: float = .75
    ):# -> tuple[Self, Any | Literal[0]]:
        """Fit the Torch Model

        Here, `data` is either a DataFrame or a PyTorch DataLoader.
        `va_idx`, is a vector or list of either integer or boolean indexing 
        those parts of the data being used for validation purposes.

        Note: if you have a separate DataLoader, for the validation data,
              you can use PyTorch's `ConcatDataset` and assign idices repectively!

        During training, the Negative Poisson LogLikelihood is used as loss function.
        This may be cahnged to a modular one, in the future.
        The loss is annealed using the gammas, from max to min!
        This is done for a fifth of all iterations to prevent early over-fitting.
        To turn this behaviour off, both `gamma_max` and `gamma_min` need to be $1$!

        Args:
            data (DataFrame | DataLoader): the whole dataset or frame.
            va_idx (iterable): _description_
            n_iter (int, optional): Max. no. training epochs. Defaults to 1000.
            batch_size (int, optional): batch size. Defaults to 250.
            verbose (int, optional): Verbose interval in steps. Defaults to 50.
            gamma_max (float, optional): Negative Poisson loss annealing T start. Defaults to 2.1.
            gamma_min (float, optional): Negative Poisson loss annealing T end. Defaults to .75.

        Returns:
            ModelWrapper: a copy of this object
            tr_loss: the final validation loss
        """
        x = data[self.features]  # type: ignore
        y = data[self.targets]  # type: ignore

        if not hasattr(self, 'opt'):
            self.opt = Adagrad(self.model.parameters(), self.lr_init)

        if isinstance(x, pd.DataFrame):
            x = x.values
        if isinstance(y, pd.DataFrame):
            y = y.values

        N, D = x.shape
        _, C = y.shape

        if not isinstance(va_idx, bool):
            idx_te = va_idx
            va_idx = np.zeros([N]).astype(bool)
            va_idx[idx_te] = True
        tr_idx = ~va_idx

        # print(x)
        x = torch.tensor(x.astype(float)).float()
        y = torch.tensor(y.astype(float)).float()

        x_va = x[va_idx]  # type: ignore
        x = Variable(x[tr_idx])  # type: ignore
        y_va = y[va_idx]  # type: ignore
        y = Variable(y[tr_idx])  # type: ignore

        gamma = np.linspace(gamma_max, gamma_min, n_iter//5)
        gamma = np.concatenate([
            gamma,
            np.ones([n_iter - n_iter//5])*gamma_min
        ], axis=0).tolist()
        for i in tqdm(range(n_iter)):

            ntr = x.shape[0]
            n = 0
            while n < ntr:
                dn = min(n + batch_size, ntr)
                y_ = self.model(x[n:dn])
                loss = nn.PoissonNLLLoss(False, True, eps=1e-10)(y_, y[n:dn]) \
                    + gamma[i] * (y_ - y[n:dn]).abs().mean()
                loss.backward()
                self.opt.step()
            with torch.no_grad():
                nte = x_va.shape[0]
                n = 0
                te_loss = 0
                while n < nte:
                    dn = min(n + batch_size, nte)
                    y_ = self.model(x_va[n:dn])
                    loss = nn.PoissonNLLLoss(
                        False, True, eps=1e-10, reduction='sum'
                    )(y_, y_va[n:dn]) / C + gamma[i](y_ - y_va).abs().sum() / C
                    te_loss += loss / nte
                if not (i+1) % verbose:
                    print(
                        f'iter {i+1}/{n_iter}: gamma = {gamma[i]}, '
                        f'Loss = {loss.detach().item() / batch_size}'
                    )

        return self, te_loss
