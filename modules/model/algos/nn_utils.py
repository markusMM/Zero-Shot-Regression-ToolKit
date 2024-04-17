from typing import Union, Iterable
import torch
from torch import nn


def compose_layer(
        module,
        activations: list = []
):
    layers = []
    layers.append(module)
    for act in activations:
        layers.append(act)
    return nn.Sequential(*layers)


def constraint_loss(
    cnstr_model: Union[object, str],
    cnstr_x_idx: Iterable[int],
    loss_fun: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    prediction: torch.Tensor,
    cnstr_weight: float = .25
):
    r"""
    Constraint loss based on some function.

    Takes a constrainer object which calculated a function, the loss is individually
    calculated upon.

    .. math::
        \xi_{final}(x, y | fx, Cx, W) = w_0 \xi(fx(x) || y) + w_1 \xi(Cx(x) || y)

    With :math:`\xi` being the loss function, Cx the constraint function / model and fx
    the actual prediction of the model of interest.

    Note that :math:`fx(x)` is a given parameter of this function, while :math:`Cx(x)` is
    calculated on the fly.

    :param cnstr_model: :math:`Cx`, constraint model object.
    :param cnstr_x_idx: Indices of x which are of interest for Cx.
    :param loss_fun: Actual loss module calculating the loss function.
    :param x: Input for the model.
    :param y: (Given) target value for the model.
    :param prediction: Prediction of the model.
    :param cnstr_weight: :math:`w_1` Weight for the constraint loss
    w.r.t. :math:´w_0 + w_1 = 1`
    :return: The final loss, :math:`\xi_{final}`
    """
    if cnstr_weight < 0:
        cnstr_weight *= -1
    while cnstr_weight > 1:
        cnstr_weight /= 100
    w_0 = 1 - cnstr_weight
    w_1 = cnstr_weight
    x = x[:, cnstr_x_idx]
    if len(x.shape) > 2:
        x = x.squeeze(-1)
    if not hasattr(cnstr_model, 'predict'):
        y_cnstr = cnstr_model(x)
    else:
        y_cnstr = cnstr_model.predict(x)
    if not isinstance(y_cnstr, torch.Tensor):
        y_cnstr = torch.tensor(y_cnstr).float()
    if y_cnstr.shape != y.shape:
        y_cnstr = y.cnstr.reshape(y.shape)
    return w_0 * loss_fun(prediction, y) + w_1 * loss_fun(prediction, y_cnstr)
