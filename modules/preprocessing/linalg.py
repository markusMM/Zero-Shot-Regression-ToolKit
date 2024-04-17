import time
import torch
import pandas as pd
from modules.preprocessing import PreProcessing
from modules.preprocessing.util import svd_low_rank_pca
from modules.log import logger


class LowRankPCACompression(PreProcessing):

    def __init__(
            self,
            columns,
            n_comp_min=None,
            n_comp_max=None,
            embed_cols=False
    ):
        """
        Low Rank PCA Reduction.

        Makes use of :meth:`~modules.preprocessing.util.svd_low_rand_pca`.
        It basically fit, based on threshold, the optimal range of principle components
        in between `n_comp_min` and `n_comp_max`, by evaluating how much variance is
        kept.

        `embed_cols` determines, if we look onto an embedding with `embed_cols` entries.
        **Then just the prefixes, without `_emb_X` need to be declared as columns.**

        :param columns: columns to be PCA compressed
        :param n_comp_min: min number of PCs
        :param n_comp_max: max number of PCs
        :param embed_cols: Whether we look a embeddings.
        """
        if isinstance(columns, str):
            columns = [columns]
        if embed_cols:
            if sum([embed_cols]) > 1:
                clip_dim = embed_cols
            else:
                clip_dim = 512
            columns = [
                f'{columns[0]}_embed_{j}'
                for j in range(clip_dim)
            ]
        super().__init__(columns)
        self.n_comp_min = n_comp_min
        self.n_comp_max = n_comp_max
        self.fitted = False
        self.n_comp_fit = n_comp_max
        self.A = None
        self.W = None
        self.E = None

    def transform(self, df: pd.DataFrame):

        # finding optimal no. components
        t = time.time_ns() * 1e-9
        col_name = '_'.join(self.columns[0].split('_')[:-1])
        logger.info(f'PCA compression for {col_name}...')

        # define data
        x = torch.tensor(df[
            self.columns
        ].fillna(0).values).float()
        x[x != x] = 0  # noqs

        # if fitted, we just need this one dimensionality
        if self.fitted:
            df = pd.concat([
                df,
                pd.DataFrame((
                    x @ self.A[:, :self.n_comp_fit]
                ).detach().numpy(), columns=[
                    f'{col_name}_pc_{j}'
                    for j in range(self.n_comp_fit)
                ], index=df.index)
            ], axis=1)
            return df

        # gathering nim and max no. of considered components nc
        # if not given try getting
        #   10 as min and
        #   55% original size as max
        # both can be
        d_orig = len(self.columns)
        nc_min = self.n_comp_min or int(d_orig and 10)
        nc_max = self.n_comp_max or int(d_orig * .55 or 1.0)

        self.A, self.W, self.E, u = svd_low_rank_pca(
            x,
            nc_max
        )
        u = pd.DataFrame(u.detach().numpy(), columns=[
            f'{col_name}_pc_{j}'
            for j in range(nc_max)
        ], index=df.index)
        df = pd.concat([df, u], axis=1)

        # loop through possible no. components nc
        eps = 1e-10  # chosen machine precision at this point
        vratio_pca_embed_opt = 0
        nc_opt = nc_min
        for nc in range(nc_min, nc_max):
            v_ratio = df[[
                f'{col_name}_pc_{j}'
                for j in range(nc)
            ]].values.var() / (df[
                self.columns
            ].fillna(0).values.var() + eps)
            if vratio_pca_embed_opt < v_ratio:
                vratio_pca_embed_opt = v_ratio
                nc_opt = nc

        df = df.drop(
            [
                f'{col_name}_pc_{j}'
                for j in range(nc_opt, nc_max)
            ], axis=1
        )

        self.fitted = True
        self.n_comp_fit = nc_opt

        dt = time.time_ns() * 1e-9 - t
        logger.info(f'done after {dt}s')
        logger.info(f'best no. components: {nc_opt}')
        logger.info(f'best variance coverage [%]: {vratio_pca_embed_opt}')

        return df
