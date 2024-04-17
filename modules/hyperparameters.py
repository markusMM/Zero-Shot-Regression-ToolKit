import collections

from modules.log import logger


class HyperParameters(collections.abc.Mapping):
    """
    Dict the hyperparameters provided in the training job.

    Allows casting of the hyperparameters
    in the `get` method.
    """

    def __init__(self, hyperparameters_dict):
        self.hyperparameters_dict = hyperparameters_dict

    def __getitem__(self, key):
        return self.hyperparameters_dict[key]

    def __len__(self):
        return len(self.hyperparameters_dict)

    def __iter__(self):
        return iter(self.hyperparameters_dict)

    def get(self, key, default=None, object_type=None):
        """
        Have the same functionality of `dict.get`.

        Allows casting of the values using the additional attribute
        `object_type`:

        Args:
            key: hyperparameter name
            default: default hyperparameter value
            object_type: type that the hyperparameter wil be casted to.

        Returns:

        """
        try:
            value = self.hyperparameters_dict[key]
            return object_type(value) if object_type else value
        except KeyError:
            logger.error(
                "Could not find the key {} "
                "in the hyperparameter dictionary "
                "we will use the default {}".format(
                    key, default
                )
            )
            return default

    def __str__(self):
        return str(self.hyperparameters_dict)

    def __repr__(self):
        return str(self.hyperparameters_dict)
