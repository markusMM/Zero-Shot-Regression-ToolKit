from modules.preprocessing.pre_processing import PreProcessing
from modules.utils import root_names
from modules.log import logger


class ConCatEmbeddings(PreProcessing):

    def __init__(
            self,
            columns: list,
            new_col: str = None,
            cut_old: bool = False
    ):
        """
        Concatenate Embeddings.

        Takes M numerical columns and concatenates them into 1 column of M dim arrays.

        :param columns: columns to concatenate
        :param new_col: new columns name, if not specified (most common name is used!)
        :param cut_old: whether to delete the original columns.
        """
        super().__init__(columns)
        if new_col is None:
            new_col = root_names(columns)
        if new_col is None:
            new_col = columns[0] + '_concat'
            logger.warn('Cannot root columns name and none was given!')
            logger.warn(f'Using {new_col}.')
        self.new_col = new_col
        self.cut_old = cut_old

    def transform(self, df):
        df[self.new_col] = list(map(lambda i: df.loc[
            i, self.columns
        ].values, df.index))
        if self.cut_old:
            df = df.drop(self.columns, axis=1)
        return df
