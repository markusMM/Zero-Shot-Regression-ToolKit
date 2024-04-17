import time
import PIL
import cv2
from typing import Union, List, Optional, Callable
import numpy as np
import pandas as pd
import torch
from torch import nn
from imutils.video import FileVideoStream
from imutils.video import FPS
import torchvision.transforms as tr
import torchvision.transforms.functional as tf
from modules.log import logger
from modules.data_retrieval.media_retrieval import download_image
from modules.preprocessing.pre_processing import PreProcessing
import open_clip
from tqdm import tqdm
from PIL import PngImagePlugin

LARGE_ENOUGH_NUMBER = 2400
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

# very slow for deployment, as it always downloads CLIP model on start!
(CLIP_MODEL, _, CLIP_PREPROCESS) = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k'
)  # laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
TOKENIZER = open_clip.get_tokenizer('ViT-B-32')


if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


def extract_frames_imutils(url, freq_frames=1, transform=None):
    """
    Extract frames from a video file using imutils.

    Extracts frames from a video file located at the specified URL using :any:`imutils`.

    The steps are the following:

    - every `freq_frames` are extracted
    - extracted frames are converted to PyTorch tensors
    - the `transform` function is applied to each frame after extraction, if given

    NOTE: This function requires the imutils, cv2, and torch packages to be installed.

    Example:
        >>> frames = extract_frames_imutils('video.mp4', freq_frames=2, transform=None)
        >>> print(frames.shape)  # Output: (num_frames, height, width, channels)

    :param url: The file path or URL of the video file to extract frames from.
    :param freq_frames: The frequency of frames to extract. If set to 1, every frame will be extracted.
                         If set to 2, every second frame will be extracted, and so on. Default is 1.
    :param transform: A function/transform to be applied to each frame after extraction.
                                       This could be used for preprocessing or data augmentation purposes.
                                       Default is None.

    :return: A tensor containing the extracted frames. Usually of
        dim(num_frames, height, width, channels)

    """
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(url, transform=transform).start()
    time.sleep(1.0)

    k = 0
    frames = []
    fps = FPS().start()
    while fvs.running():
        frame = fvs.read()
        if not k % freq_frames and frame is not None:
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # noqa
            except cv2.error:  # noqa
                pass
            frames.append(torch.tensor(frame)[None])

        if fvs.Q.qsize() < 2:  # If we are low on frames, give time
            time.sleep(0.001)  # Ensures producer runs now
        fps.update()
        k += 1

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    fvs.stop()

    return torch.vstack(frames)


def scale_img(img, target_image_size):
    """
    Scales, resizes and crops an image to a target size.

    :param img: The input image to be scaled. It can be either a PIL.Image.Image object or a NumPy array.
    :type img: PIL.Image.Image or numpy.ndarray
    :param target_image_size: The target size of the output image (both width and height).
    :type target_image_size: int
    :return: The scaled and cropped image as a PyTorch tensor.
    :rtype: torch.Tensor
    """
    if type(img) is not PIL.Image.Image:  # noqa
        img = PIL.Image.fromarray(np.uint8(img))  # noqa
    s = min(img.size)
    r = target_image_size / s
    s = [round(r * img.size[1]), round(r * img.size[0])]
    img = tf.resize(img, s, interpolation=PIL.Image.LANCZOS)  # noqa
    img = tf.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(tr.ToTensor()(img), 0)
    return img


class CLIPImagePreprocessing(PreProcessing):
    """
    CLIP embedding of a image link column.

    With CLIP the Vision Transformer (ViT) embedding is meant.
    It was specifically trained to match its embedding with the one of the newest NLP
    model, based on the text description to the respective scene.

    The columns are supposed to be link lists for images.
    It automatically downloads the image in the link of the resp. column with Pillow!
    Every fail in digitalizing an image from the resp. entrance is ignored (NaN).

    If `grouper` is set, the incoming data is always grouped by that column, if able.
    This process will then automatically only select the first instance.
    Grouped processing will then be merged with the original data.

    :param columns: the columns to be extracted and embedded.
    :param normalize: whether to normalize the final embedding.
    :param grouper: whether to group the preprocessing and on which column.
    """

    def __init__(self, columns: List[str], normalize: bool = True, grouper: str = None):
        super().__init__(columns)  # noqa
        self.normalize = normalize
        self.grouper = grouper

    def transform_image(self, image: List[str], col: str):  # noqa
        if image is not None:  # noqa
            if not isinstance(image, list):  # noqa
                image = [image]  # noqa
            if len(image) == 0:  # noqa
                return pd.DataFrame(None, index=[0])  # noqa
            img = None
            for im in image:  # noqa
                img_ = download_image(im)  # noqa
                if img is not None and image != np.NaN:  # noqa
                    if np.prod(img.size) < np.prod(img_.size):  # noqa
                        img = img_  # noqa
                else:  # noqa
                    img = img_  # noqa
            if img is None:  # noqa
                return pd.DataFrame(None, index=[0])  # noqa
            with torch.no_grad():
                image_features = CLIP_MODEL.encode_image(
                    CLIP_PREPROCESS(img).unsqueeze(0)
                )
                if self.normalize:
                    image_features /= image_features.norm(dim=-1, keepdim=True)
            return pd.DataFrame(
                image_features.detach().numpy(),
                columns=[f"{col}_clip_{j}" for j in range(image_features.shape[1])],
            )
        return pd.DataFrame(None, index=[0])

    def transform(self, df):
        for c in self.columns:
            embedding = []
            if self.grouper is not None and self.grouper not in df:
                logger.warn(f'cannot find grouper "{self.grouper}" in data!')
                self.grouper = None
            if self.grouper is None:
                for im in tqdm(df[c]):
                    embedding.append(self.transform_image(im, c))
                new_df = pd.concat(embedding, axis=0).reset_index(drop=True)
                df = pd.concat([df.copy().reset_index(drop=True), new_df], axis=1)
            else:
                clip_col = pd.concat(list(map(
                    lambda d: pd.concat([
                        self.transform_image(d[1][c], c),
                        pd.DataFrame([[d[1][self.grouper]]], columns=[self.grouper])
                    ], axis=1),
                    df.groupby(self.grouper)[
                        c
                    ].first().reset_index().iterrows()
                )))
                df = df[list(
                    set(df.columns).difference(clip_col.columns)
                ) + [self.grouper]]
                df = df.merge(clip_col, on=self.grouper)
        return df


def gaussian_filter(kernel_size, sigma=1, muu=0):
    """
    Generates a 2D Gaussian filter kernel.

    :param kernel_size: The size of the kernel (both width and height).
    :type kernel_size: int
    :param sigma: The standard deviation of the Gaussian distribution. Default is 1.
    :type sigma: float
    :param mu: The mean of the Gaussian distribution. Default is 0.
    :type mu: float
    :return: The Gaussian filter kernel.
    :rtype: numpy.ndarray
    """
    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
    x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
    dst = np.sqrt(x ** 2 + y ** 2)  # distance
    normal = 1 / (2.0 * np.pi * sigma ** 2)  # lower normalization

    return np.exp(-((dst - muu) ** 2 / (2.0 * sigma ** 2))) * normal


class Embedding2dMap(PreProcessing):
    """
    Markus' Recursive Hilbert-Peano fcn
    it can break down from any dxd square
    which is not a multiple of a prime number
    it has an asymetric 3x3 base in case of a
    hilbert structure before.
    curve.
    -> A completely centered solution for square 2D space fillings.
    @author: Markus Meister
    """

    def __init__(
        self,
        columns: List[List[str]],
        kernel_size: int = 256,
        stride: int = 128,
        padding: int = 64,
    ):
        super(Embedding2dMap, self).__init__(columns)
        assert np.round(np.sqrt(kernel_size)) ** 2 == kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ksize2d = int(np.sqrt(kernel_size))
        self.step2d = int(np.sqrt(stride))
        self.map2d = self.hilbert_peano(self.ksize2d)
        self.window = gaussian_filter(self.ksize2d)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        rep2d = []
        for c in self.columns:
            assert type(c) is list
            if len(c) < 2:
                c = c[0]
            emb2d = pd.DataFrame(
                [list(map(
                    lambda x:
                    self.span_ova(self.window_seq(torch.tensor(x))).detach().numpy(),
                    df[c].values
                ))]
            )
            emb2d.columns = [
                "_".join(c[0].split("_")[:-1]) + f"_2d_{j}"
                for j in range(emb2d.shape[1])
            ]
            rep2d.append(emb2d)
        rep2d = pd.concat(rep2d, axis=1)
        return pd.concat([
            df.reset_index(drop=True).iloc[:, 1:],
            rep2d.reset_index(drop=True).iloc[:, 1:]
        ], axis=1)

    def window_seq(self, x: Union[np.ndarray, torch.Tensor]):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        if self.padding > 0:
            x = torch.stack(
                [torch.zeros(self.padding), x, torch.zeros(self.padding)], dim=0
            )
        return x[None].unfold(-1, self.kernel_size, self.stride)

    def span_ova(self, x):
        x = (x[..., self.map2d] * self.window).squeeze()  # is [k, d, d]
        d_ova = self.ksize2d + self.step2d * (x.shape[-3] - 1)
        k = x.shape[0]
        roll_ids = torch.cat(
            list(map(lambda h: torch.arange(k).roll(-h, 0)[None], range(k))), 0
        )
        folding = nn.Fold(d_ova, (self.ksize2d, self.ksize2d), stride=self.step2d)
        return folding(x[roll_ids].reshape(1, k * k, -1).permute(0, 2, 1))

    @staticmethod
    def prime_factors(n):
        i = 2
        factors = []
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors.append(i)
        if n > 1:
            factors.append(n)
        return factors

    def prime_check(self, n):
        primes = self.prime_factors(n)
        mod = True
        for prime in primes:
            if all(np.mod(prime, [2, 3])):
                mod = False
        return mod

    def hilbert_peano(
        self, d: int, verbose: bool = False, sub: bool = False
    ) -> np.ndarray:

        curve_map_2d = np.zeros([d, d], dtype=int)
        hil = not np.ceil(np.mod(d, 2)) or d == 2  # noqa
        if hil:
            bas = np.array([[2, 1], [3, 4]], dtype=int).T - 1
        else:
            bas = np.flipud(np.array([[1, 6, 7], [2, 5, 8], [3, 4, 9]]).T.tr.tr) - 1  # noqa
        sub_hilm3 = np.array([[5, 6, 7], [4, 3, 8], [1, 2, 9]], dtype=int) - 1
        diff = d // (3 - hil)
        # print(diff)
        hm3 = not np.ceil(np.mod(diff, 2))
        if not hil and sub:  # noqa
            bas = sub_hilm3
        if not np.mod(d, 3 - hil) and diff > 1:  # noqa
            for r in bas.flatten():
                # print(r)
                m = self.hilbert_peano(
                    diff or d // (2 + hm3), verbose=verbose, sub=hil and not hm3 or sub  # noqa
                )  # noqa

                if not hm3 and hil and not np.mod((r + 1), 2):
                    m = np.fliplr(np.max(m) - m)

                if hil:
                    if r == 0:
                        m = np.fliplr(np.fliplr(m).T)
                    if r == 3:
                        m = m.T
                else:
                    if not np.mod((r + 1), 2):
                        m = np.flipud(m)
                    if 3 < r + 1 < 7:
                        m = np.fliplr(np.flipud(m))

                ind_x, ind_y = np.where(bas == r)
                ind_y += 1
                ind_y *= diff
                ind_x += 1
                ind_x *= diff
                if verbose:
                    print(ind_x, ind_y)
                    print(diff)
                    print(m)
                curve_map_2d[ind_x[0] - diff: ind_x[0], ind_y[0] - diff: ind_y[0]] = (
                    m + r * diff ** 2
                )
        else:
            if sub and hil:
                curve_map_2d = sub_hilm3
            else:
                curve_map_2d = bas
        return curve_map_2d


class CLIPTextPreprocessing(PreProcessing):
    """
    Preprocess text data for CLIP embeddings.

    CLIPTextPreprocessing applies the specified CLIP (Contrastive Language-Image Pretraining) ViT model's
    text encoding to text entries in the input DataFrame. It processes each text element in the specified
    columns individually. Note that the preprocessing is performed element-wise, but batch processing may
    be implemented in the future.

    CLIP represents the artificial synthesis between text and image encoding, enabling the encoding of
    textual information into 512 double-precision values. Each text entry is automatically encoded into
    an array of 512 numbers. By default, these arrays are normalized by dividing by the vector norm,
    leading to improved numerical stability and further processing capabilities.

    :param columns: A list of column names containing text data to be preprocessed.
    :type columns: List[str]
    :param avg: A flag indicating whether to average embeddings if there are multiple entries in one item.
                Defaults to True.
    :type avg: bool, optional
    :param ppf: An optional post-processing function to be applied after encoding, aggregation, and scaling.
                Defaults to None.
    :type ppf: Optional[Callable], optional

    Note:
    - The CLIPTextPreprocessing class requires the use of the CLIP model for text encoding and the associated
      tokenizer.
    - If the input text is represented as a list, each element of the list is treated as a separate text entry.
    - The preprocessing pipeline includes text encoding, optional averaging of embeddings, and post-processing
      using an optional function.
    """

    def __init__(
        self,
        columns: List[str],
        avg: bool = True,
        ppf: Optional[Callable] = None
    ):
        super().__init__(columns)
        self.avg = avg
        self.ppf = ppf

    def transform_text(self, text: Union[str, list], col: str):
        if isinstance(text, list):
            if len(text) < 1:
                text = None
            else:
                if not isinstance(text[0], str):
                    text = None
        if isinstance(text, float) or text is not None:
            try:
                with torch.no_grad():
                    tokenized = TOKENIZER(text)
                    text_features = CLIP_MODEL.encode_text(tokenized)
                    if len(text_features.shape) > 1 and any(np.array(list(
                        set(text_features.shape).difference([512])
                    )) > 0):
                        dim_id = np.where(
                            np.array(text_features.shape) != 512
                        )[0].tolist()
                        if self.avg:
                            text_features = text_features.mean(
                                axis=dim_id, keepdim=True
                            )
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                return pd.DataFrame(
                    text_features.detach().numpy().squeeze()[None],
                    columns=[f"{col}_clip_{j}" for j in range(text_features.shape[-1])],
                )
            except TypeError:
                logger.debug(
                    f'text {text} from columns {col} was unable to be converted!'
                )
        return pd.DataFrame(None, index=[0])

    def transform(self, df):
        for c in self.columns:
            embedding = []
            for tx in tqdm(df[c]):
                embedding.append(self.transform_text(tx, c))
            new_df = pd.concat(embedding, axis=0).reset_index(drop=True)
            df = pd.concat([df.reset_index(drop=True), new_df], axis=1)
        return df


class GroupedTextListCLIP(PreProcessing):
    """
    Preprocess text data grouped by a specified column for CLIP embeddings.

    GroupedTextListCLIP applies the specified CLIP (Contrastive Language-Image Pretraining) ViT model's
    text encoding to text entries in the input DataFrame, grouped by a specified column.

    NOTE: There is always just the first element, in the group, used for transformation!

    CLIP represents the artificial synthesis between text and image encoding, enabling the encoding of
    textual information into 512 double-precision values. Each text entry is automatically encoded into
    an array of 512 numbers. By default, these arrays are normalized by dividing by the vector norm,
    leading to improved numerical stability and further processing capabilities.

    NOTE: The GroupedTextListCLIP class requires the use of the CLIP model for text encoding and the associated
      tokenizer.

    NOTE: The preprocessing pipeline includes text encoding, grouping by a specified column, and merging
      the embeddings with the original DataFrame.

    :param columns: A list of column names containing text data to be preprocessed.
    :type columns: List[str]
    :param grouper: The column name used for grouping text entries for efficient processing.
                    Defaults to 'creative_id'.
    :type grouper: str, optional
    """

    def __init__(
            self,
            columns,
            grouper='creative_id'
    ):
        super().__init__(columns)
        self.grouper = grouper

    def transform(self, df: pd.DataFrame):

        def encode_texts(row, col):
            x = row[col]
            row = pd.DataFrame(row).T
            nodata = pd.DataFrame(
                index=row.index,
                columns=[f'{col}_clip_{j}' for j in range(512)]
            )
            if x is None:
                return pd.concat([row, nodata])
            if x != x:  # noqa
                return pd.concat([row, nodata])
            df = pd.DataFrame(
                CLIP_MODEL.encode_text(
                    TOKENIZER(x)
                ).float().mean(0).numpy()[None].tolist(),
                columns=[f'{col}_clip_{j}' for j in range(512)],
                index=row.index
            )
            return pd.concat([row, df], axis=1)
        for col in self.columns:
            with torch.no_grad(), torch.autocast(DEVICE):
                clip_col = pd.concat(list(map(
                    lambda d: encode_texts(d[1], col),
                    df.groupby(self.grouper)[
                        col
                    ].first().reset_index().iterrows()
                ))).drop(col, axis=1)
            df = df.merge(clip_col, on=self.grouper)

        return df


class GroupedInsertNameHereListCLIPAVG(PreProcessing):
    """Load and average CLIP for countries."""

    def __init__(self, columns: List[str], mode='country', grouper: str = None):
        super().__init__(columns=columns)
        self.grouper = grouper
        if isinstance(mode, str):
            self.reftab = globals().get(mode.upper() + '_DF')
            assert self.reftab != None
        elif isinstance(mode, pd.DataFrame):
            self.reftab = mode
        elif isinstance(mode, dict):
            self.reftab = pd.DataFrame(mode)

        assert len(list(filter(lambda c: 'clip' in c, self.reftab.columns))) > 0
        if mode in self.reftab:
            self.rcol = mode
        else:
            all_clip_candidates = self.reftab.columns[
                self.reftab.columns.str.contains('_clip_\.+')  # noqa
            ].str.replace('_clip_\.+', '').unique()  # noqa
            rcols_filtered = list(filter(
                lambda c: 'clip' not in c and 'code' not in c and '_id' not in c,
                self.reftab.columns
            ))
            rcol_candidates = sum(map(
                lambda cand:
                rcol_candidates[
                    pd.Series(rcols_filtered).str.contains(cand)
                ],
                all_clip_candidates
            ))
            if len(rcol_candidates) <= 0:
                rcol_candidates = all_clip_candidates
            self.rcol = rcol_candidates[0]

    def avg_embeddings(self, ref_list, group, col) -> pd.DataFrame:
        rcol = self.rcol
        reftab = self.reftab
        if isinstance(ref_list, str):
            try:
                ref_list = eval(ref_list)  # noqa
            except (ValueError, NameError, SyntaxError):
                ref_list = [ref_list]
        clip_cols = reftab.columns.str.contains('clip_\d+')  # noqa
        clip_suff = '_clip_' + reftab.columns[clip_cols].str.replace(
            '.+_' * reftab.columns[clip_cols][0].count('_'),
            ''
        )
        if ref_list is None or ref_list != ref_list:  # noqa
            df_clip = pd.DataFrame(
                np.ones((1, sum(clip_cols))) * np.NaN,
                columns=(col+clip_suff).tolist()
            )
        else:
            try:
                df_clip = pd.DataFrame(np.nanmean(np.concatenate(list(map(
                    lambda c: reftab.loc[
                        reftab[rcol] == c, clip_cols
                    ].values.reshape(-1, 512),
                    ref_list
                )), axis=0), axis=0).reshape(1, 512), columns=(col+clip_suff).tolist())
            except Exception as e:  # noqa
                logger.warn(str(e))
                logger.warn(f'cannot convert {self.rcol} entries: \n{ref_list}')
                df_clip = pd.DataFrame(
                    np.ones((1, sum(clip_cols))) * np.NaN,
                    columns=(col+clip_suff).tolist()
                )
        df_clip.columns = [f'{col}_clip_{k}' for k in range(sum(clip_cols))]
        df_clip[self.grouper] = group
        return df_clip

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in self.columns:
            embeddings = list(map(
                lambda d: self.avg_embeddings(d[1][c], d[1][self.grouper], c),
                df.groupby(self.grouper).first().reset_index().iterrows()
            ))
            if len(embeddings):
                embeddings = pd.concat(embeddings)
                df = df[list(
                    set(df.columns).difference(embeddings.columns)
                ) + [self.grouper]].merge(
                    embeddings,
                    on=self.grouper
                )

        return df


class CLIPAverager(PreProcessing):
    """
    Preprocess CLIP embeddings by averaging across specified columns.

    Averages CLIP embeddings across specified columns in the input
    DataFrame across each embedding dimension and normalizes the result
    to improve numerical stability.

    :param columns: A list of column names containing CLIP embeddings to be averaged.
    :type columns: List[str]
    :param colname: The name prefix of the new columns that will store the averaged
                    CLIP embeddings.
                    If not provided, a default name will be used.
                    Defaults to None.
    :type colname: str, optional
    """

    def __init__(self, columns: list, colname: str = None):
        super().__init__(columns)
        self.colname = colname

    def transform(self, df: pd.DataFrame):
        clip_arr = np.zeros((len(self.columns), len(df), 512))
        for nc, col in enumerate(self.columns):
            if 'clip' not in col:
                col = col + '_clip'
            cols = list(filter(lambda c: col in c, df.columns))
            if len(cols) != 512:
                logger.warn(f'Could not fetch {col}!')
                continue
            clip_arr[nc] = df[cols].fillna(0)
        clip_arr = np.nanmean(clip_arr, axis=0)
        clip_arr /= np.sqrt(np.nansum(clip_arr**2, axis=1)[:, None])
        cols = list(map(lambda c: c.replace('_clip', ''), self.columns))
        if self.colname is None:
            self.colname = "_".join(cols)
        df[[
            f'{self.colname}__avg_clip_{c}'
            for c in range(clip_arr.shape[1])
        ]] = clip_arr
        return df.copy()
