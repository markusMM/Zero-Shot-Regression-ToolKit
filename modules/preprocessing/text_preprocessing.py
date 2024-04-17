import json
import logging

import numpy as np
import pandas as pd
from modules.preprocessing.creative_encoding import CLIPTextPreprocessing
from modules.preprocessing.pre_processing import PreProcessing


def flatten_native_texts(row, extract_types=["title", "description", "call_to_action"]):
    """
        Flatten and extract native text data from a row in a DataFrame.

        Extracts native text data from a row in a DataFrame and flattens it into a
        single DataFrame row from the specified `extract_types` fields.

        NOTE: The input `row` needs to have the following two fields:

        - 'native_text_types' - a list of names of the texts
        - 'native_texts' - a list of the actual texts

        Parameters:
        :param row: A pandas Series representing a row in a DataFrame containing native text data.
                    (pd.Series)
        :param extract_types: A list of native text types to extract, such as 'title', 'description', or 'call_to_action'.
                               Default is ['title', 'description', 'call_to_action'].
                               (List[str])

        Returns:
        :return: A DataFrame containing extracted native text data flattened into a single row.
                 The columns correspond to the specified extract types.
                 (pd.DataFrame)

        Example:

        >>> row = pd.Series({
        ...     'native_texts': '["Buy Now", "Great deals"]',
        ...     'native_text_types': '["call_to_action", "description"]'
        ... })
        >>> extracted_text = flatten_native_texts(row)
        >>> print(extracted_text)
        ... "Buy Now. Great deals."
        """
    extract_types = pd.Series(extract_types).str.replace("native_", "")
    col_names = "native_" + pd.Series(extract_types)
    if row.isna().values.sum():
        return pd.DataFrame([[""] * len(extract_types)], columns=col_names)
    if isinstance(row["native_texts"], str):
        row["native_text_types"] = json.loads(row["native_text_types"])
        row["native_texts"] = json.loads(row["native_texts"])
    native_text_types = np.array(row["native_text_types"])
    native_texts = np.array(row["native_texts"])

    row = pd.DataFrame(index=[0])
    for et in extract_types:
        row["native_" + et] = ""
        row["native_" + et] = ". ".join(native_texts[native_text_types == et])
    row.columns = col_names
    return row  # .drop(['native_text_types', 'native_texts'], axis=1)


class ExtractTextsPreProcessing(PreProcessing):
    """
    Extract and flatten text data from specified columns in a DataFrame.

    Used to extract and flatten text data from specified columns in a DataFrame.
    :meth:`~modules.preprocessing.text_preprocessing.flatten_native_texts`

    By default, it concatenates "title", "description" and "call_to_action" from the
    text info. But with `extract_types` this can be changed.

    NOTE: The input `df` needs to have the following two fields:

    - 'native_text_types' - lists of names of the texts
    - 'native_texts' - lists of the actual texts

    Parameters:
    :param columns: A list of column names containing text data to be extracted and flattened.
                    (List[str])
    :param extract_types: List of keywords within the textx to extracted.

    Methods:
    :return: The DataFrame with extracted and flattened text data concatenated to the original columns.
             (pd.DataFrame)

    Example:

    >>> processor = ExtractTextsPreProcessing(columns=['text_column'])
    >>> df_processed = processor.transform(df)
    >>> print(df_processed[].head())
    ... "Extracted texts. Here we go!."
    """
    def __init__(self, columns, extract_types):
        super().__init__(columns)
        self.extract_types = extract_types

    def fit(self, df):
        return self

    def fit_transform(self, df):
        return self.transform(df)

    def transform(self, df):
        df_txts = pd.concat(
            list(
                map(
                    lambda x:
                    flatten_native_texts(
                        df.loc[x, self.columns], self.extract_types
                    ),
                    df.index
                )
            ),
            axis=0,
        )
        for c in df_txts.columns:
            df[c] = df_txts[c].values
        return df


class CombTextEmbPreProcessing(PreProcessing):
    """
    Combine and preprocess text data from multiple columns for text embedding.

    The CombTextEmbPreProcessing class is used to combine and preprocess text data
    from multiple columns in a DataFrame to prepare it for text embedding using CLIP model.

    Parameters:
    :param columns: A list of column names containing text data to be combined and preprocessed.
                    (List[str])
    :param txt_col_name: The name of the column to store the combined text data.
                         Default is "texts".
                         (str)

    Methods:
    :return: The DataFrame with combined and preprocessed text data ready for text embedding.
             (pd.DataFrame)

    Example:

    >>> processor = CombTextEmbPreProcessing(columns=['title', 'description'])
    >>> df_processed = processor.transform(df)
    >>> print(df_processed[f'texts_clip_{j}' for j in range(512)].head(1).values[:6])
    ... array([-4.31843235e-01, -5.05673925e-01,  1.28450912e+00, -2.56871827e+00,
       -3.45429184e-01,  9.80903116e-01, -7.01629551e-01,  8.71225669e-01,
       -1.10112869e-01,  1.15748972e+00, -1.09570521e+00,  1.39876149e+00])
    """
    def __init__(self, columns, txt_col_name="texts"):
        super().__init__(columns)
        self.txt_col_name = txt_col_name

    def transform(self, df: pd.DataFrame):
        """
        Transform the DataFrame by combining and preprocessing text data from specified columns.

        This method combines text data from multiple columns into a single column, preprocesses the text data,
        and prepares it for text embedding using the CLIP model.

        :param df: The DataFrame containing text data to be transformed.
                   (pd.DataFrame)
        :return: The transformed DataFrame with combined and preprocessed text data ready for text embedding.
                 (pd.DataFrame)
        """
        tcn = self.txt_col_name
        df[tcn] = ""
        for c in self.columns:
            if c not in df:
                try:
                    df = ExtractTextsPreProcessing([c]).transform(df)
                    df[tcn] += ". ".join(df[c])
                except KeyError:
                    continue
        return CLIPTextPreprocessing([tcn]).transform(df)
