import os
import pandas as pd
from modules.preprocessing.mappers import Mapper
from modules.data_retrieval.df_handlers import save_unstring_all
CD = '/'.join(__file__.replace('\\', '/').split('/')[:-1])
TR_DATA_PTH = os.path.join(CD, '../test_data')


class TestMapper:

    pyload = save_unstring_all(
        pd.read_csv(TR_DATA_PTH + '/tr_data.csv')
    )

    def test_transform(self):
        req = self.pyload
        ppmp = Mapper(['f_24'], {'f_24': {'native': 1, 'video': 2, 'banner': 3}})
        df = ppmp.transform(req)
        assert df['f_24'].map(lambda x: (1 in x) or (2 in x)
                              or (3 in x)).sum().item() < 1
