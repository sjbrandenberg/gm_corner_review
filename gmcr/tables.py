"""Column types for the per-trace tables written to parquet."""

import numpy as np
import pandas as pd

_FLAG_TEXT = {"True": True, "False": False}


def _missing(v):
    return v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v))


def parquet_safe(df):
    """Make df's non-numeric columns writable to parquet, in place, and return it.

    A column can mix types (a flag that is missing for traces that failed extraction, or typed
    differently in different files). True/False columns, including ones already written as the text
    'True'/'False', become nullable booleans, so a flag still reads as a flag after a round trip
    (bool("False") is True). Every other non-numeric column becomes text.
    """
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            continue
        present = col[~col.map(_missing)]
        if len(present) and present.map(lambda v: isinstance(v, (bool, np.bool_)) or v in _FLAG_TEXT).all():
            df[c] = pd.array([None if _missing(v) else _FLAG_TEXT.get(v, v) for v in col], dtype="boolean")
        elif col.dtype == object:
            df[c] = col.map(lambda v: None if _missing(v) else str(v))
    return df
