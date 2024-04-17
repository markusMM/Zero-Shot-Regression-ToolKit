from typing import Callable, Union, Iterable
import torch
from torch import nn
from sklearn.metrics import r2_score as r2
from pytorch_lightning import (
    LightningModule
)
from torch.optim import AdamW
from modules.utils import get_obj
from modules.loss_functions.torch_losses import RMSLELoss
from modules.model.algos.layers import ADBUDGResponse
from modules.loss_functions.torch_outlier_detectors import std_threshold
from modules.model.algos.nn_utils import compose_layer, constraint_loss


class MLP(LightningModule):
    """
    Multi-Layer Perceptron

    This is a generic class for any feed-forward neural network (NN).
    The main architecture it comes with is a Multi-Layer Perceptron (MLP).

    An MLP usually has just an input->hidden->output layers.
    The units from one layer to the next are usually all-to-all connected.

    Each layer can have an activation function (here ReLU... Rectified Linear Unit).

    The default MLP of this class has additional input and last hidden ADNs
    (Activation, Dropout, Norm).
    This class already supports the following layers for this:
        - ADBUDG
        - Sigmoid
        - Dropout
        - Bernoulli Dropout
        - Batch-Norm

    Additionally one can declare to build a full-skipnet instead of an MLP.
    Then a Residual MLP is build from this parent module.
    It concatenates all inputs of the previous layer to the input of the current layer,
    in a weighted fashion. (The input and output of the previous layer are weighted.)

    :param n_input: # input units
    :param n_output: # output units
    :param n_hidden: list of #s of hidden units (ordered)
    :param optimizer: optimizer for training. default AdamW.
    :param lr_init: initial learning rate
    :param loss: loss function for optimization. Default MSELoss
    :param seed: random number seed. Default 42
    :param use_variable_adbudg: Wheather to use ADBUDG for input variables
    :param use_variable_dropout: Whether (and optionally with which intensity) to use
    input Dropout. Default is False. (Default intensity for True is 0.6)
    :param use_variable_batchnorm: whether to have input Batch-Norm
    :param use_response_adbudg: whether to use last hidden ADBUDG
    :param use_response_dropout: whether to use last layer Dropout.
    Default is False. (Default intensity for True is 0.6)
    :param use_response_batchnorm: Whether to use last hidden Batchnorm
    :param use_response_sigmoid: Whether to use last hidden Sigmoid.
    :param use_mid_relu: Whether to use mid-layer ReLU
    :param non_neg_response: Whether to use last hidden ReLU
    :param skip_net: Whether to use a Skip-Net architecture.
    :param nn_arch: Optional: Any given architecture with arguments n_inout, n_hidden
    and n_output
    :param scheduler: Learning rate scheduler. Default Cosine Annealing
    :param scheduler_params: Scheduler parameters
    :param constraint_loss_model: A model which calculates an alternate prediction the
    model output is additionally compared to.  Default is :code:`None`
    :param constraint_weight: Ratio for constraint loss if :code:`constraint_loss_model`
    is given. Default is :math:`0.25`
    :param validation_string: Passphrase to identify validation metric in training
    :param cut_outliers: Whether to cut outliers mid-training
    :param out_fun: Outlier identification function, needs to output the indices.
    """
    def __init__(
            self,
            n_input: int,
            n_output: int,
            n_hidden: list = [256],
            optimizer=None,
            lr_init: float = 1e-3,
            loss=RMSLELoss(),
            seed: int = 42,
            use_variable_adbudg: bool = False,
            use_variable_dropout: bool = False,
            use_variable_batchnorm: bool = True,
            use_response_adbudg: bool = False,
            use_response_dropout: bool = False,
            use_response_batchnorm: bool = True,
            use_response_sigmoid: bool = True,
            use_mid_relu: bool = True,
            non_neg_response: bool = True,
            skip_net: bool = False,
            nn_arch: nn.Module = None,
            scheduler: type = None,
            scheduler_params: dict = None,
            constraint_loss_model: Union[object, str] = None,
            constraint_x_idx: Iterable[int] = None,
            constraint_weight: float = .25,
            validation_string: str = 'val_loss',
            cut_outliers: bool = False,
            out_fun: Callable = std_threshold
    ):
        super().__init__()

        if scheduler_params is None:
            scheduler_params = {}

        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau

        if optimizer is None:
            self.optimizer = AdamW

        self.save_hyperparameters()

        if nn_arch is None and not skip_net:
            self.layers = []
            n_units = [n_input] + n_hidden + [n_output]
            if use_variable_adbudg:
                self.layers.append(
                    ADBUDGResponse(n_units[0])
                )
            if use_variable_dropout:
                if isinstance(use_variable_dropout, float):
                    n_var = use_variable_dropout
                else:
                    n_var = .6
                self.layers.append(
                    nn.Dropout(n_var)
                )
            if use_variable_batchnorm:
                self.layers.append(
                    nn.BatchNorm1d(n_units[0])
                )
            for n in range(len(n_units) - 2):
                self.layers.append(
                    nn.Linear(n_units[n], n_units[n + 1])
                )
                self.layers.append(
                    nn.BatchNorm1d(n_units[n + 1])
                )
                if use_mid_relu:
                    self.layers.append(nn.ReLU())
            # last-last layer before regression
            if use_response_batchnorm:
                self.layers.append(
                    nn.BatchNorm1d(n_units[-2])
                )
            if use_response_dropout:
                if isinstance(use_response_dropout, float):
                    n_var = use_response_dropout
                else:
                    n_var = .6
                self.layers.append(
                    nn.Dropout(n_var)
                )
            if use_response_adbudg:
                self.layers.append(
                    ADBUDGResponse(n_units[-2])
                )
            elif use_response_sigmoid:
                self.layers.append(
                    nn.Sigmoid()
                )
            # final regression layer
            self.layers.append(
                nn.Linear(n_units[-2], n_units[-1])
            )
            if non_neg_response:
                self.layers.append(
                    nn.ReLU()
                )
            self.layers = nn.Sequential(*self.layers)
        elif skip_net:
            self.layers = SkipNet(
                n_input,
                n_output,
                n_hidden,
                use_variable_adbudg,
                use_variable_dropout,
                use_variable_batchnorm,
                use_response_adbudg,
                use_response_dropout,
                use_response_batchnorm,
                use_response_sigmoid,
                use_mid_relu,
                non_neg_response
            )
        else:
            self.layers = nn_arch(
                n_input,
                n_output,
                n_hidden
            )

        # parse constraint model
        if constraint_loss_model is not None:
            if constraint_x_idx is None:
                constraint_x_idx = 0
            if isinstance(constraint_loss_model, str):
                constraint_loss_model = get_obj(constraint_loss_model)

        self.n_input = n_input
        self.n_output = n_output
        self.num_loss = loss.__class__(reduction='none')
        self.learning_rate = lr_init
        self.scheduler = scheduler
        self.scheduler_params = scheduler_params
        self.seed = seed
        self.loss = loss
        self.constraint_loss_model = constraint_loss_model
        self.constraint_x_idx = constraint_x_idx
        self.constraint_weight = constraint_weight
        self.validation_string = validation_string
        self.valid_losses = []
        self.valid_accuracies = []
        self.cut_outliers = cut_outliers
        self.out_fun = out_fun
        self.outliers = None
        self.opt = None
        self.scd = None

    def valid_loss(self):
        return torch.stack(self.valid_losses).mean()

    def valid_accuracy(self):
        return torch.stack(self.valid_accuracies).mean()

    def detect_outliers(self, batch):
        # batch vise average training loss
        with torch.no_grad():
            x, y = batch
            y_hat = self.layers(x)
            diffs = self.num_loss(
                y_hat, y
            ).reshape(-1, self.n_output).mean(-1)
        return self.out_fun(diffs)

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        # batch
        x, y = batch
        # batch vise outlier discrimination
        if self.cut_outliers and self.out_fun is not None:
            outliers = self.detect_outliers(batch)
            if outliers is not None:
                if len(outliers) == len(y):
                    x = x[~outliers]
                    y = y[~outliers]
                self.outliers = outliers
        # loss
        y_hat = self.layers(x)
        loss = self.calc_loss(x, y, y_hat)
        self.log('train_loss', loss)
        return loss

    def calc_loss(self, x, y, y_hat):
        if self.constraint_loss_model is None:
            return self.loss(y, y_hat)
        else:
            return constraint_loss(
                self.constraint_loss_model,
                cnstr_x_idx=self.constraint_x_idx,
                loss_fun=self.loss,
                x=x,
                y=y,
                prediction=y_hat,
                cnstr_weight=self.constraint_weight
            )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        loss = self.calc_loss(x, y, y_hat)
        accuracy = r2(
            y_hat.detach().numpy(),
            y.detach().numpy()
        )
        self.valid_losses.append(loss*len(y))
        self.valid_accuracies.append(torch.tensor(accuracy))
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self):
        avg_loss = self.valid_loss()
        avg_acc = self.valid_accuracy()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)

    def configure_optimizers(self):
        if self.opt is None:
            self.opt = self.optimizer(self.parameters(), lr=self.learning_rate)
        self.scd = self.scheduler(
            self.opt, **self.scheduler_params
        )
        optimizer_dict = {
            "optimizer": self.opt,
            "lr_scheduler": self.scd,
            "monitor": self.validation_string
        }
        return optimizer_dict


class SkipNet(nn.Module):
    """
    SkipNet

    This variant of an MLP uses skip connections for each layer, consequently.
    This basically means that the input of the previous layer is concatenated to their
    output and weighted. This will not apply to the last layer as it is not connected to
    any further layer!

    :param n_in: # input values
    :param n_out: # uotput values
    :param n_hidden: # hidden units in between I and O
    :param use_variable_adbudg: Wheather to use ADBUDG for input variables
    :param use_variable_dropout: Whether (and optionally with which intensity) to use
    input Dropout. Default is False. (Default intensity for True is 0.6)
    :param use_variable_batchnorm: whether to have input Batch-Norm
    :param use_response_adbudg: whether to use last hidden ADBUDG
    :param use_response_dropout: whether to use last layer Dropout.
    Default is False. (Default intensity for True is 0.6)
    :param use_response_batchnorm: Whether to use last hidden Batchnorm
    :param use_response_sigmoid: Whether to use last hidden Sigmoid.
    :param use_mid_relu: Whether to use mid-layer ReLU
    :param non_neg_response: Wether to use last hidden ReLU
    """
    def __init__(
            self,
            n_in,
            n_out,
            n_hidden,
            use_variable_adbudg: bool = False,
            use_variable_dropout: bool = False,
            use_variable_batchnorm: bool = True,
            use_response_adbudg: bool = False,
            use_response_dropout: bool = False,
            use_response_batchnorm: bool = True,
            use_response_sigmoid: bool = True,
            use_mid_relu: bool = True,
            non_neg_response: bool = True,
    ):
        super().__init__()
        n_units = [n_in] + n_hidden + [n_out]
        self.weights = [
            nn.Parameter(5*torch.randn(2))
        ] * (len(n_units) - 1)

        # input ADN
        adn = []
        if use_variable_adbudg:
            adn.append(
                ADBUDGResponse(n_units[0])
            )
        if use_variable_dropout:
            if isinstance(use_variable_dropout, float):
                n_var = use_variable_dropout
            else:
                n_var = .6
            adn.append(
                nn.Dropout(n_var)
            )
        if use_variable_batchnorm:
            adn.append(
                nn.BatchNorm1d(n_units[0])
            )
        if len(adn) > 0:
            self.in_adn = nn.Sequential(*adn)
        else:
            self.in_adn = None

        # first layer
        self.in_layer = compose_layer(
            nn.Linear(n_units[0], n_units[1]),
            [nn.ReLU()] if use_mid_relu else []
        )

        self.layers = []
        # hidden layers
        for l in range(1, len(n_units) - 2):
            self.layers.append(compose_layer(
                nn.Linear(n_units[l] + n_units[l - 1], n_units[l + 1]),
                [nn.ReLU()] if use_mid_relu else []
            ))


        hid_adn = []
        # last-last layer before regression
        if use_response_batchnorm:
            hid_adn.append(
                nn.BatchNorm1d(n_units[-2] + n_units[-3])
            )
        if use_response_dropout:
            if isinstance(use_response_dropout, float):
                n_var = use_response_dropout
            else:
                n_var = .6
            hid_adn.append(
                nn.Dropout(n_var)
            )
        if use_response_adbudg:
            hid_adn.append(
                ADBUDGResponse(n_units[-2] + n_units[-3])
            )
        elif use_response_sigmoid:
            hid_adn.append(
                nn.Sigmoid()
            )
        if len(hid_adn):
            self.hid_adn = nn.Sequential(*hid_adn)
        else:
            self.hid_adn = None

        # final regression layer
        self.out_layer = compose_layer(
            nn.Linear(n_units[-2] + n_units[-3], n_units[-1]),
            [nn.ReLU()] if non_neg_response else []
        )

        self.n_input = n_in
        self.n_output = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.in_adn is not None:
            x = self.in_adn(x)
        h = self.in_layer(x)

        # skip loop over all layers
        # always concatenates the outputs of the last and the last last layers as input
        # for the current one.
        for l, layer in enumerate(self.layers):
            h_temp = h
            h = layer(torch.cat([
                self.weights[l][0] * x,
                self.weights[l][1] * h
            ], axis=1))
            x = h_temp

        h = torch.cat([
                self.weights[l + 1][0] * x,
                self.weights[l + 1][1] * h
        ], axis=1)
        if self.hid_adn is not None:
            h = self.hid_adn(h)

        return self.out_layer(h)
