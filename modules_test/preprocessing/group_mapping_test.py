import pandas as pd
from modules.preprocessing.group_mapping import GroupedPreProc


class TestGroupedPreProc:
    def test_transform(self):
        # Sample input DataFrame
        df = pd.DataFrame({
            'group_column': ['A', 'A', 'B', 'B'],
            'column1': [1, 2, 3, 4],
            'column2': [5, 6, 7, 8]
        })
        class MockPreProc:
            def __init__(self, columns):
                self.columns = columns

            def transform(self, df):
                df['transformed_column'] = df[self.columns[0]] * 2
                return df

        pp_class = MockPreProc
        pp_params = {'columns': ['column1']}
        grouped_preproc = GroupedPreProc(
            columns=['group_column'],
            pp_class=pp_class,
            pp_params=pp_params
        )

        transformed_df = grouped_preproc.transform(df)

        expected_output = pd.DataFrame({
            'group_column': ['A', 'A', 'B', 'B'],
            'column1': [1, 2, 3, 4],
            'column2': [5, 6, 7, 8],
            'transformed_column': [2, 2, 6, 6]
        })

        pd.testing.assert_frame_equal(transformed_df, expected_output)
