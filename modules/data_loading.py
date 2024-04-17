import glob
from abc import ABC
from itertools import chain
from typing import Optional, Union, Callable, List, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from modules.common import NCPU
from modules.transforms import fill_na


class CSVDataSet(IterableDataset, ABC):
    valid = False
    xcols_final = None

    def __init__(
        self,
        csvs_path: str,
        x_cols: Optional[Union[str, List[str]]],
        y_cols: Union[str, List[str]],
        validation_size: int = 10,
        processor: Optional[Callable] = None,
        shuffle_csv_order: bool = True,
    ):
        super().__init__()
        # getting labels
        self.x_cols = x_cols
        self.y_cols = y_cols

        # parse processor
        if processor is None:

            def identity(x):
                return x

            processor = identity
        self.validation_size = validation_size
        self.processor = processor

        # list csv files in path
        self.csvs = glob.glob(csvs_path + "/**/*.csv", recursive=True)
        if shuffle_csv_order:  # random shuffle? (y|n):
            np.random.shuffle(self.csvs)

        # reset current status
        self._reset_()

    def get_validation_set(self):
        """Get `self.validation_size` rows."""
        self._reset_()
        self.valid = True
        return self.get_nrows(self.validation_size)

    def get_nrows(self, nrow):
        """Get N rows."""
        df = []  # the datarame from the rows
        while len(df) < nrow:
            d = self._get_next_row_()
            if d is None:
                break
            df.append(d)
        if len(df) <= 0:
            return False
        else:
            df = pd.concat(df, axis=0)  # concatenation of all rows
            return self._prepare_xy_(df)  # processing all rows

    def _get_next_row_(self):
        """Get a row"""
        try:  # avoid iteration over empty rows
            df = self.full_reader.__next__()
            assert df is not None
            return df
        except Exception:  # noqa
            return None

    def _reset_(self):
        """
        Reset the iterable.

        NOTE: Skips the first `self.validation_size` rows if set.
        """
        self.readers = list(
            map(lambda c: pd.read_csv(c, chunksize=1, iterator=True), self.csvs)
        )
        self.full_reader = chain(*self.readers)
        if self.valid:
            for k in range(self.validation_size):
                self.full_reader.__next__()

    def _prepare_xy_(self, df: pd.DataFrame) -> Tuple[torch.TensorType]:
        """
        Process feature columns `x` and label column(s) `y` into tensors.


        Note:
            it uses `self.processor` to process the features `x`
            and `self.x_cols` and `self.y_cols` for drawing the columns.

        input:
            df: `pd.DataFrame`
                The input data.
        output:
            (x,y): `Tuple` of `torch.TensorType`
                The processed feature `x` and label `y` columns.

        """
        # processing
        df = self.processor(df)

        # sanity check
        df.fillna(0)
        df.replace(np.inf, np.finfo(np.float16).max)

        # parsing columns
        if isinstance(self.y_cols, str):
            self.y_cols = [self.y_cols]
        if self.x_cols is None:
            self.x_cols = list(set(df.columns).difference(self.y_cols))
        else:
            self.x_cols = list(filter(lambda c: c in df.columns, self.x_cols))

        # get final x columns
        self.xcols_final = self.x_cols

        # final return of tensors
        x = torch.tensor(np.array(df[self.x_cols]))
        y = torch.tensor(np.array(df[self.y_cols]))
        return x, y

    @staticmethod
    def __prep_xy_iter__(x: torch.TensorType, y: torch.TensorType):
        """Get rid of batch dimension for iterating over the data."""
        return x.squeeze(0).float(), y.squeeze(0).float()

    def get_xcols(self):
        if self.xcols_final is None:
            self._reset_()
            self.get_nrows(1)
            self._reset_()
        return self.xcols_final

    def __iter__(self):
        """Construct and send iterable to `torch.util.data.DataLoader`."""
        # reinitialize the readers
        self._reset_()
        # return a map of the preparing function
        return iter(
            map(
                lambda x: self.__prep_xy_iter__(*self._prepare_xy_(x)), self.full_reader
            )
        )


def generate_dataloader(
        data, xcols, ycols, indices=None,
        batch_size=256, num_workers=NCPU,
        shuffle=True, nan_transform=fill_na
):
    if isinstance(data, pd.DataFrame):
        if indices is None:
            indices = data.index
        dataset = TensorDataset(
            torch.FloatTensor(nan_transform(
                data.loc[indices, xcols].astype(float).values
            )),
            torch.FloatTensor(nan_transform(
                data.loc[indices, ycols].astype(float).values
            ))
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
        )
    elif isinstance(data, str):
        dataset = CSVDataSet(
            csvs_path=data,
            x_cols=xcols,
            y_cols=ycols,
            validation_size=0,
            processor=None,
            shuffle_csv_order=shuffle
        )
        shuffle = False
        return DataLoader(dataset, batch_size=batch_size)

    return DataLoader(data, batch_size=batch_size)
