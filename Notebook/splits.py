"""Walk-forward splits shared by every model in the project (MS2 notebook, §6.7).

Expanding window: training always starts in SAMPLE_START and grows by one year per fold.
A 1-month embargo separates the last training target month from the first evaluation month.

mode='tune'  -> one fold per validation year 2011..2015 (choose hyperparameters here, then freeze them)
mode='test'  -> annual refit: one fold per test year 2016..2021 (test ends 2021-11), same frozen hyperparameters

fold_zscore(df, cols, f.train_idx) z-scores columns with one fold's training rows (§6.5).

Usage:
    from splits import walk_forward_splits
    for f in walk_forward_splits(df, mode='tune'):
        model.fit(X.iloc[f.train_idx], y.iloc[f.train_idx])
        pred = model.predict(X.iloc[f.eval_idx])
"""
from typing import Iterator, NamedTuple

import numpy as np
import pandas as pd

SAMPLE_START = pd.Period('1980-01', 'M')   # first training target month
TUNE_YEARS = range(2011, 2016)             # validation years (VAL_END = 2015-12)
TEST_YEARS = range(2016, 2022)             # test years, refit before each
TEST_END = pd.Period('2021-11', 'M')       # last target month with a target (GKX ends 2021-12)
EMBARGO_MONTHS = 1


class Fold(NamedTuple):
    name: str                 # e.g. 'tune-2011', 'test-2016'
    train_idx: np.ndarray     # positional row indices (use with .iloc / numpy)
    eval_idx: np.ndarray
    train_start: pd.Period
    train_end: pd.Period      # last training target month
    eval_start: pd.Period
    eval_end: pd.Period


def walk_forward_splits(df: pd.DataFrame, mode: str = 'tune', date_col: str = 'target_month',
                        start: pd.Period = SAMPLE_START, embargo: int = EMBARGO_MONTHS) -> Iterator[Fold]:
    """Yield expanding-window folds over the rows of df, keyed on the target month in date_col."""
    if mode not in ('tune', 'test'):
        raise ValueError("mode must be 'tune' or 'test'")
    years = TUNE_YEARS if mode == 'tune' else TEST_YEARS
    tm = pd.PeriodIndex(df[date_col], freq='M') if not isinstance(df[date_col].dtype, pd.PeriodDtype) \
        else pd.PeriodIndex(df[date_col])
    for year in years:
        eval_start = pd.Period(f'{year}-01', 'M')
        eval_end = min(pd.Period(f'{year}-12', 'M'), TEST_END)
        train_end = eval_start - 1 - embargo
        train = (tm >= start) & (tm <= train_end)
        ev = (tm >= eval_start) & (tm <= eval_end)
        fold = Fold(f'{mode}-{year}', np.flatnonzero(train), np.flatnonzero(ev),
                    start, train_end, eval_start, eval_end)
        # guards: no overlap, embargo respected, tuning never touches the test window
        assert not np.intersect1d(fold.train_idx, fold.eval_idx).size
        assert tm[fold.train_idx].max() <= eval_start - 1 - embargo if fold.train_idx.size else True
        if mode == 'tune':
            assert eval_end <= pd.Period(f'{TUNE_YEARS[-1]}-12', 'M')
        yield fold


def describe_splits(df: pd.DataFrame, date_col: str = 'target_month') -> pd.DataFrame:
    """One row per fold in both modes: date ranges and row counts."""
    rows = []
    for mode in ('tune', 'test'):
        for f in walk_forward_splits(df, mode, date_col):
            rows.append({'fold': f.name, 'train': f'{f.train_start} .. {f.train_end}',
                         'embargo': str(f.train_end + 1), 'evaluate': f'{f.eval_start} .. {f.eval_end}',
                         'train rows': len(f.train_idx), 'eval rows': len(f.eval_idx)})
    return pd.DataFrame(rows).set_index('fold')


def fold_zscore(df: pd.DataFrame, cols, train_idx, missing_cols=None):
    """Z-score `cols` with the mean and std of ONE fold's training rows (refit at every walk-forward step, §3.6).

    Missing values become 0 (= the training mean) and get a 0/1 `<col>_missing` indicator. By default the
    indicators cover the columns with any missing value in df: missingness here is structural (series that
    start late), not something learned from the evaluation rows.
    Returns (Z, stats): Z has df's index; stats holds the training mean and std of each column.
    """
    X = df[list(cols)].astype(float)
    tr = X.iloc[train_idx]
    mu, sd = tr.mean(), tr.std(ddof=0)
    sd = sd.where(sd > 0, 1.0)                                   # constant or all-missing column: no division by 0
    Z = ((X - mu) / sd).fillna(0.0)
    miss = [c for c in cols if X[c].isna().any()] if missing_cols is None else list(missing_cols)
    Z = pd.concat([Z, X[miss].isna().astype('int8').add_suffix('_missing')], axis=1)
    return Z, pd.DataFrame({'mean': mu, 'std': sd})
