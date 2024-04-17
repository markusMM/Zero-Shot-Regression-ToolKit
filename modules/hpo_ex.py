import joblib
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from modules import NCPU
from modules.log import logger


class LightRayHPO:

    def __init__(
        self,
        model_class,
        model_params: dict,
        param_spaces: dict = dict(
            lr_init=tune.loguniform(1e-2, .2),
            n_hidden=tune.choice([
                [400, 800]
                [1800],
                [800, 1200],
                [2400],
                [1000, 2000],
                [3200],
                [1400, 2600],
                [4000]
            ]),
            burn_in=tune.randint(0, 200),
            T_max=tune.randint(200, 500)
        ),
        name: str = 'HPO_imps_mlp_01',
        n_sample: int = 220
    ):
        for k in param_spaces.keys():
            if k in model_params:
                del model_params[k]

        self.model_class = model_class
        self.model_params = model_params
        self.param_spaces = param_spaces
        self.name = name,
        self.n_samples = n_sample

    def fit(self, data_train, data_valid=None):

        def step(params, train_data, valid_data=None):
            model = self.model_class(
                **self.model_params,
                **params,
                callbacks=[TuneReportCallback(
                    metrics=dict(
                        loss="ptl/val_loss",
                        accuracy="ptl/val_accuracy"
                    ),
                    on="validation_end"
                )]
            )
            model.fit(train_data, data_valid=valid_data)

        trainable = tune.with_parameters(
            step,
            data_train, data_valid=data_valid
        )
        analysis = tune.run(
            trainable,
            resources_per_trial={
                "cpu": NCPU
            },
            metric="loss",
            mode="min",
            config=self.param_spaces,
            num_samples=self.n_samples,
            name=f"tune_{self.name}"
        )

        joblib.dump(analysis, f'{self.name}.pkl')

        logger.info(analysis.best_config)
