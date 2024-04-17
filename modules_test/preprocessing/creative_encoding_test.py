import glob
import pytest
import pandas as pd
import numpy as np
import torch
from PIL import Image
from modules.utils import glob_file
from modules.preprocessing.creative_encoding import (
    Embedding2dMap,
    CLIPImagePreprocessing,
    extract_frames_imutils,
    scale_img
)


class TestEmbedding2dMap:
    df = pd.DataFrame(
        np.ones((80, 16 + 4 * (3 - 1))),
        columns="rnd_clip_" + pd.Series(np.arange(16 + 4 * (3 - 1)).astype(str)),
    )

    @pytest.mark.dependency(name="test:mapped:shape&numbers")
    def test_shape(self):
        self.prep = Embedding2dMap([[self.df.columns.tolist()]], 16, 4, 0)
        self.prep.window = torch.ones(4, 4)
        self.df = self.prep.transform(self.df)
        assert self.df.shape[0] == 80
        assert self.df.iloc[0, -1].shape == (1, 1, 8, 8)
        assert (
            self.df.iloc[0, -1]
            != torch.tensor(
                [
                    [
                        [
                            [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                            [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                            [2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0],
                            [2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0],
                            [2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0],
                            [2.0, 2.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0],
                            [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                            [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                        ]
                    ]
                ]
            )
        )


class TestCLIPImagePreProcessing:

    pic_path = glob.glob(
        "modules_test/test_data/images/image.png"
    ) + glob.glob(
        "../test_data/images/image.png"
    )
    df = pd.DataFrame(
        [[pic_path]],
        columns=["images"],
    )
    dfg = pd.DataFrame(
        [[pic_path, 31]]*4 + [[np.NaN, 0]]*4,
        columns=['img', 'cid']
    )

    @pytest.mark.dependency(name="clip:embed")
    def test_embedding(self):
        self.prep = CLIPImagePreprocessing(["images"])
        nc = self.df.shape[1]
        self.df = self.prep.transform(self.df)
        assert self.df.shape[1] == nc + 512
        assert self.df.columns[-1] == "images_clip_511"

    @pytest.mark.dependency(name="clip:group")
    def test_grouping(self):
        df = CLIPImagePreprocessing(["img"], grouper='cid').transform(self.dfg)
        assert sum(map(lambda c: 'clip' in c, df.columns)) == 512
        ccol = list(filter(lambda c: 'clip' in c, df.columns))[0]
        assert len(df) == len(self.dfg)
        assert sum(~df[ccol].isna()) == 4
        assert sum(df[ccol].isna()) == 4


class TestExtractFramesImutils:

    vi = (glob.glob(
        "modules_test/test_data/videos/test_video.mp4"
    ) + glob.glob(
        "../test_data/videos/test_video.mp4"
    ))[0]

    @pytest.mark.dependency(name="extract:frames")
    def test_function(self):

        frames_1 = extract_frames_imutils(self.vi, 16)
        frames_2 = extract_frames_imutils(self.vi, 32)

        assert np.round(len(frames_1) / len(frames_2)) == 2


class TestScaleImg:

    im = (glob.glob(
        "modules_test/test_data/images/image.png"
    ) + glob.glob(
        "../test_data/images/image.png"
    ))[0]

    @pytest.mark.dependency(name="scale:img")
    def test_sizes(self):
        x = scale_img(Image.open(self.im), 64)
        assert x.shape[-2:] == (64, 64)

        x = scale_img(Image.open(self.im), 2048)
        assert x.shape[-2:] == (2048, 2048)
