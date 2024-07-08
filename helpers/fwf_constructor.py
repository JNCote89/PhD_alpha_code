import numpy as np


class FWFConstructor:

    def __init__(self, column_starts: list[int], last_column_lenght: int, column_names: list[str]):
        self.column_starts = column_starts
        self.last_column_lenght = last_column_lenght
        self.column_names = column_names

    @property
    def column_specs(self) -> list[tuple[int, int]]:
        # Metadata usually starts at 1, but the index start at zero...
        first_column_start = self.column_starts[0]
        fwf_columns_specs = []

        for index, col_start in enumerate(self.column_starts):
            if col_start == self.column_starts[-1]:
                fwf_columns_specs.append(tuple(np.subtract((col_start, col_start + self.last_column_lenght),
                                                           first_column_start).tolist()))
            else:
                fwf_columns_specs.append(
                    tuple(np.subtract((col_start, self.column_starts[index + 1]), first_column_start).tolist()))
        return fwf_columns_specs

    @property
    def column_names(self) -> list[str]:
        return self._column_names

    @column_names.setter
    def column_names(self, value):
        if len(value) != len(self.column_specs):
            raise ValueError(f"The column specs does not match the column names! \n"
                             f"col spec = {len(self.column_specs)} \n"
                             f"col names = {len(value)}")
        else:
            self._column_names = value


