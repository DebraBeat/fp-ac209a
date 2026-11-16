# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -widgets,-colab,-language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # CS1090A / AC209A — Milestone 2: Data Wrangling and Project Redefinition
#
# **Canvas project group #: __** *(fill in)* · **Project #12 — Forecasting Stock Volatility on the S&P 500: Can Machine Learning Beat HAR?**
#
# **Group members:** Debbie · YiTong Li · Cathy · Anthony · Nic
#
# | § | Section | Owner | Status in this notebook |
# |---|---|---|---|
# | 0 | Setup | YiTong | code |
# | 1 | Getting the data | **Debbie** (collected the data) — loading code by YiTong | code ✔ |
# | 2 | Understanding the data (rows, columns, types, summary stats, **timing check**) | **YiTong** | code ✔ |
# | 3 | Cleaning & preprocessing of the stock panel (universe, target draft, missingness, unusual values, scale, imbalance, leakage) | **YiTong** | code ✔ |
# | 4 | Time-series EDA (distribution, persistence, stationarity, auto-ARIMA, per-stock HAR, clustering, cross-section, seasonality) | **YiTong** | code ✔ |
# | 5 | Target & volatility features (final target, HAR/EWMA/vol-of-vol features, leakage test, jump check, baselines + QLIKE) | **Cathy** | code ✔ |
# | 6 | Market data integration & leakage audit (Goyal–Welch macro, FF5, merged table, walk-forward splits) | **Anthony** | *to do* (text) |
# | 7 | Sector mapping, data dictionary, revised proposal & slides | **Nic** | *to do* (text) |
# | 8 | Revised project question (draft for the group) | all | text |
#
# > **One-line summary.** One row of our final table is *one large U.S. stock in one month*. From everything known at the end of month *t−1*, we predict the log of that stock's realized volatility in month *t*. The questions are whether pooled ML models (ridge/lasso/elastic net, gradient-boosted trees) beat the per-stock HAR benchmark, which features drive any gain, and whether the gain survives in stress periods.
#
# > **Why monthly, not weekly (change from the original proposal).** The data the group has is the Gu–Kelly–Xiu (GKX) monthly firm-characteristics panel, plus the monthly Goyal–Welch macro data and Fama–French factors. None of them has daily high/low prices, so the weekly Garman–Klass target cannot be built from them. GKX does include `retvol`, the standard deviation of a stock's daily returns within a month, which is a monthly realized volatility. We therefore move the forecast horizon from one week to one month, with HAR lags of 1, 3 and 12 months instead of 1, 5 and 22 days.

# %% [markdown]
# ## 0. Setup

# %%
import os, time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60); pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: f'{v:,.4g}')
plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 13, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10})
# ---- Colab: mount Google Drive; final_project = Debbie's shared folder, added as a shortcut in My Drive ----
import zipfile, io
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA = '/content/drive/MyDrive/final_project'          # raw data
    OUT  = '/content/drive/MyDrive/209_ms2_outputs'        # cleaned tables -> your own Drive (survives runtime resets)
except ImportError:                                         # running locally (not Colab)
    DATA, OUT = os.path.join('..', 'Data'), 'data_clean'
assert os.path.isdir(DATA), f'{DATA} not found: add final_project as a shortcut to My Drive first'
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(1090)
# Walk-forward split fixed IN ADVANCE (target months). EDA that informs model choices uses <= VAL_END only.
TRAIN_END, VAL_END = pd.Timestamp('2010-12-31'), pd.Timestamp('2015-12-31')
STRESS_EPISODES = [('1973-11','1974-12','Oil shock'), ('1987-10','1987-12','1987 crash'),
                   ('2000-09','2002-10','Dot-com bust'), ('2008-09','2009-06','GFC'),
                   ('2020-02','2020-06','COVID')]


# %% [markdown]
# ## 1. Getting the data  *(owner: Debbie — data collected; loading code: YiTong)*
#
# | File (in `../Data`) | Source | Frequency / span | Used for |
# |---|---|---|---|
# | `datashare.csv` (raw, 3.8 GB) → `datashare_small.parquet` (same rows, float32) | Gu, Kelly & Xiu (2020) firm characteristics, https://dachxiu.chicagobooth.edu/download/datashare.zip | monthly, 1957-01 → 2021-12, all CRSP stocks | stock-level features **and** the volatility target (`retvol`) |
# | `datashare_sample.parquet` | 700 random stocks from the same file | same | quick prototyping only |
# | `Data2024.xlsx - Monthly.csv` | Goyal & Welch predictor data (2024 update), https://sites.google.com/view/agoyal145 | monthly, 1871 → 2024-12 | market-wide state (S&P variance `svar`, T-bill, spreads, valuation ratios) |
# | `F-F_Research_Data_5_Factors_2x3.csv` | Kenneth French Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html | monthly, 1963-07 → 2026-09 | market/size/value/profitability/investment factor returns |
#
# The raw files are already on disk (downloaded by Debbie; the original zips are in `../Data/Archive`). The cell below **reports their size** and loads them.

# %%
def folder_sizes(root):
    rows = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f); rows.append((os.path.relpath(p, root), os.path.getsize(p) / 2**20))
    return pd.DataFrame(rows, columns=['file', 'MB']).sort_values('MB', ascending=False)
sizes = folder_sizes(DATA)
display(sizes)
print(f'Total size of the data folder: {sizes.MB.sum()/1024:.2f} GB')
# on Drive the 3.6 GB raw csv is stored only zipped: read its uncompressed size from the zip header (no extraction)
GKX_ZIP = next(os.path.join(dp, f) for dp, _, fs in os.walk(DATA) for f in fs if f == 'datashare.zip')
with zipfile.ZipFile(GKX_ZIP) as z:
    GKX_CSV = [n for n in z.namelist() if n.endswith('.csv')][0]
    print(f'raw GKX csv inside datashare.zip: {z.getinfo(GKX_CSV).file_size/2**30:.2f} GB uncompressed '
          f'({z.getinfo(GKX_CSV).compress_size/2**30:.2f} GB zipped)')

# %%
t0 = time.time()
gkx = pd.read_parquet(os.path.join(DATA, 'datashare_small.parquet'))
gkx['date'] = pd.to_datetime(gkx['date'])
# the parquet must contain every row of the raw csv (count newlines without parsing 3.8 GB)
CHECK_ROWS = True                     # streams 3.6 GB out of the zip (a few minutes on Drive); False to skip
if CHECK_ROWS:
    with zipfile.ZipFile(GKX_ZIP) as z, z.open(GKX_CSV) as fh:
        n_csv = sum(chunk.count(b'\n') for chunk in iter(lambda: fh.read(1 << 24), b'')) - 1
else:
    n_csv = len(gkx)
print(f'GKX parquet: {gkx.shape[0]:,} rows x {gkx.shape[1]} cols; raw csv rows = {n_csv:,}; '
      f'match = {n_csv == len(gkx)}; in-memory {gkx.memory_usage().sum()/2**30:.2f} GB; {time.time()-t0:.0f}s')

gw = pd.read_csv(os.path.join(DATA, 'Data2024.xlsx - Monthly.csv'))
gw['ym'] = pd.PeriodIndex(gw['yyyymm'].astype(str), freq='M')
gw = gw.set_index('ym').drop(columns='yyyymm').apply(pd.to_numeric, errors='coerce')
print(f'Goyal-Welch: {gw.shape[0]:,} months x {gw.shape[1]} cols, {gw.index.min()} -> {gw.index.max()}')

def read_ff_monthly(path):
    rows = []
    if path.endswith('.zip'):                            # on Drive the factor file ships as a zip
        with zipfile.ZipFile(path) as z:
            lines = io.TextIOWrapper(z.open(z.namelist()[0]), encoding='latin-1').readlines()
    else:
        lines = open(path, encoding='latin-1').readlines()
    for line in lines:
        parts = [p.strip() for p in line.split(',')]
        if 'Annual' in line: break                      # monthly block ends where the annual block begins
        if len(parts) == 7 and parts[0].isdigit() and len(parts[0]) == 6:
            rows.append(parts)
    ff = pd.DataFrame(rows, columns=['yyyymm', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    ff.index = pd.PeriodIndex(ff.pop('yyyymm'), freq='M')
    return ff.astype(float) / 100                        # percent -> decimal
FF_FILE = next(f for f in ['F-F_Research_Data_5_Factors_2x3.csv', 'F-F_Research_Data_5_Factors_2x3_CSV.zip']
               if os.path.exists(os.path.join(DATA, f)))
ff = read_ff_monthly(os.path.join(DATA, FF_FILE))
print(f'Fama-French 5: {ff.shape[0]:,} months x {ff.shape[1]} cols, {ff.index.min()} -> {ff.index.max()}')

# %% [markdown]
# ## 2. Understanding the data  *(owner: YiTong)*
#
# ### 2.1 What one row is
# * **GKX panel:** one row = **one stock (`permno`, CRSP's permanent ID) at one month-end (`date`)**. Each row has 94 firm characteristics plus the 2-digit SIC industry code `sic2`.
# * **Goyal–Welch:** one row = one month of market-wide variables.
# * **Fama–French:** one row = one month of five factor returns plus the risk-free rate.
#
# The key fact for leakage is *which month* each value in a GKX row describes. The GKX readme says characteristics have already been lagged, so that a row dated *t* can be used to predict the return of month *t*. Section 2.4 checks this against outside data.
#
# **The three identifier columns of the GKX file** (every other column is a characteristic, defined in the dictionary below):
#
# | Column | Plain meaning | Technical meaning | Example |
# |---|---|---|---|
# | `permno` | The stock's "ID card number". It never changes, even if the company changes its name or ticker. | CRSP *PERMNO*, an integer that permanently identifies a security in the CRSP database (Center for Research in Security Prices). Our join and grouping key for "the same stock over time". | `10107` = Microsoft |
# | `date` | The month the row belongs to. | Last **trading** day of the calendar month (e.g. 1957-03-29), parsed from `DATE` (YYYYMMDD). Because it is the last trading day and not the 31st/30th, we key months on the *calendar month*, not the exact day (see §3.1). | `2020-03-31` |
# | `sic2` | Which industry the company is in. | First two digits of the Standard Industrial Classification code. Stored as a float, but **categorical**: 73 is not "more" than 28. | `73` = business services / software |
#
# #### Glossary of terms used throughout this notebook
# *Each term is defined in plain words first, then technically.*
#
# * **Stock-month**: one stock observed in one month, i.e. one row of the GKX panel. *Technically:* a (`permno`, month) pair; the panel has 4.1 M of them.
# * **Panel data**: many series (stocks) observed over many periods (months) in one long table. *Technically:* long format, indexed by (`permno`, `date`); not i.i.d., because rows of the same stock and of the same month are correlated.
# * **Cross-section / cross-sectional**: "all stocks in the same month". A cross-sectional median is the median over all stocks in one month. *Technically:* `groupby('date')` operations.
# * **Volatility**: how much a price jumps around. A stock that moves ±3% a day is more volatile than one that moves ±0.5%. *Technically:* the standard deviation of returns; **realized volatility** is the standard deviation measured from the returns that actually happened in a period (here: daily returns within one month = GKX `retvol`). **Realized variance (RV)** is its square.
# * **Annualized volatility**: daily volatility rescaled to a yearly number that investors quote (e.g. "20% a year"). *Technically:* daily std × √252 (252 trading days per year).
# * **Log volatility** (`lrv`, target `y`): the natural log of `retvol`. Logs turn "×2" moves into "+0.69" moves, so a crash month no longer dwarfs every other month. *Technically:* `np.log(retvol)`; a typical large stock has `retvol` ≈ 0.018 (1.8% a day), so log ≈ −4.0.
# * **Target / response (`y`)**: the number we try to forecast: the log realized volatility of a stock in month *t* (built in §3.2).
# * **Feature / predictor**: any input the model may use to make the forecast, known *before* the month being forecast.
# * **HAR (Heterogeneous AutoRegressive) model**: "next month's volatility ≈ a weighted mix of last month's, the last quarter's and the last year's volatility". *Technically:* OLS of $y_t$ on `har_1`, `har_3`, `har_12` (Corsi 2009), the standard volatility benchmark; defined precisely in §3.2.
# * **Universe**: the set of stocks we study in a given month. Ours = the 500 largest stocks by market value that month (§3.3), a stand-in for the S&P 500.
# * **Leakage (information leakage)**: accidentally letting the model see information from the future (or from the answer itself) at training time. It makes a model look great in testing and fail in real use.
# * **Walk-forward split / embargo**: train on the past, validate on the next block of time, test on the block after; never shuffle time. An *embargo* is an unused gap between blocks so that no training row overlaps a test row.
# * **Stress month**: a month in which the whole market was turbulent; defined in §3.7 as S&P 500 realized variance (`svar`) above the 90th percentile of the training period.

# %%
# data dictionary for the 94 GKX characteristics (Green, Hand & Zhang 2017 / GKX 2020 Table A.6)
# freq = how often the underlying data updates: M monthly (CRSP prices), Q quarterly / A annual (Compustat)
D = [
('absacc','Absolute accruals','accounting','A'),('acc','Working capital accruals','accounting','A'),
('aeavol','Abnormal earnings announcement volume','trading','Q'),('age','Years since first Compustat coverage','firm','A'),
('agr','Asset growth','accounting','A'),('baspread','Bid-ask spread','trading','M'),('beta','Market beta (3y weekly)','risk','M'),
('betasq','Beta squared','risk','M'),('bm','Book-to-market','valuation','A'),('bm_ia','Industry-adjusted book-to-market','valuation','A'),
('cash','Cash holdings / assets','accounting','Q'),('cashdebt','Cash flow to debt','accounting','A'),('cashpr','Cash productivity','accounting','A'),
('cfp','Cash flow to price','valuation','A'),('cfp_ia','Industry-adjusted cash flow to price','valuation','A'),
('chatoia','Industry-adj. change in asset turnover','accounting','A'),('chcsho','Change in shares outstanding','accounting','A'),
('chempia','Industry-adj. change in employees','accounting','A'),('chinv','Change in inventory','accounting','A'),
('chmom','Change in 6-month momentum','momentum','M'),('chpmia','Industry-adj. change in profit margin','accounting','A'),
('chtx','Change in tax expense','accounting','Q'),('cinvest','Corporate investment','accounting','Q'),
('convind','Convertible debt indicator (0/1)','accounting','A'),('currat','Current ratio','accounting','A'),
('depr','Depreciation / PP&E','accounting','A'),('divi','Dividend initiation (0/1)','accounting','A'),('divo','Dividend omission (0/1)','accounting','A'),
('dolvol','log dollar trading volume','trading','M'),('dy','Dividend yield','valuation','A'),('ear','Earnings announcement return','trading','Q'),
('egr','Growth in common equity','accounting','A'),('ep','Earnings to price','valuation','A'),('gma','Gross profitability','accounting','A'),
('grcapx','Growth in capital expenditures','accounting','A'),('grltnoa','Growth in long-term net operating assets','accounting','A'),
('herf','Industry sales concentration (Herfindahl)','firm','A'),('hire','Employee growth rate','accounting','A'),
('idiovol','Idiosyncratic return volatility (3y weekly)','risk','M'),('ill','Amihud illiquidity','trading','M'),
('indmom','Industry momentum','momentum','M'),('invest','Capex and inventory investment','accounting','A'),('lev','Leverage','accounting','A'),
('lgr','Growth in long-term debt','accounting','A'),('maxret','Maximum daily return in the month','risk','M'),
('mom12m','12-month momentum','momentum','M'),('mom1m','1-month momentum (last month return)','momentum','M'),
('mom36m','36-month momentum','momentum','M'),('mom6m','6-month momentum','momentum','M'),('ms','Financial statement score (Mohanram, 0-8)','accounting','Q'),
('mvel1','Market equity (size, $000), lagged','firm','M'),('mve_ia','Industry-adjusted size','firm','A'),
('nincr','Number of consecutive earnings increases (0-8)','accounting','Q'),('operprof','Operating profitability','accounting','A'),
('orgcap','Organizational capital','accounting','A'),('pchcapx_ia','Industry-adj. % change in capex','accounting','A'),
('pchcurrat','% change in current ratio','accounting','A'),('pchdepr','% change in depreciation','accounting','A'),
('pchgm_pchsale','% chg gross margin - % chg sales','accounting','A'),('pchquick','% change in quick ratio','accounting','A'),
('pchsale_pchinvt','% chg sales - % chg inventory','accounting','A'),('pchsale_pchrect','% chg sales - % chg receivables','accounting','A'),
('pchsale_pchxsga','% chg sales - % chg SG&A','accounting','A'),('pchsaleinv','% change in sales-to-inventory','accounting','A'),
('pctacc','Percent accruals','accounting','A'),('pricedelay','Price delay (speed of reaction to market news)','trading','M'),
('ps','Piotroski F-score (0-9)','accounting','A'),('quick','Quick ratio','accounting','A'),('rd','R&D increase (0/1)','accounting','A'),
('rd_mve','R&D to market cap','valuation','A'),('rd_sale','R&D to sales','accounting','A'),('realestate','Real estate holdings','accounting','A'),
('retvol','Std. dev. of daily returns in the month (realized volatility)','risk','M'),('roaq','Return on assets','accounting','Q'),
('roavol','Earnings volatility','accounting','Q'),('roeq','Return on equity','accounting','Q'),('roic','Return on invested capital','accounting','A'),
('rsup','Revenue surprise','accounting','Q'),('salecash','Sales to cash','accounting','A'),('saleinv','Sales to inventory','accounting','A'),
('salerec','Sales to receivables','accounting','A'),('secured','Secured debt','accounting','A'),('securedind','Secured debt indicator (0/1)','accounting','A'),
('sgr','Sales growth','accounting','A'),('sin','Sin stock: alcohol/tobacco/gaming (0/1)','firm','A'),('sp','Sales to price','valuation','A'),
('std_dolvol','Volatility of dollar volume','trading','M'),('std_turn','Volatility of share turnover','trading','M'),
('stdacc','Accrual volatility','accounting','Q'),('stdcf','Cash flow volatility','accounting','Q'),('tang','Tangibility / debt capacity','accounting','A'),
('tb','Tax income to book income','accounting','A'),('turn','Share turnover','trading','M'),('zerotrade','Zero-trading days','trading','M')]
dd = pd.DataFrame(D, columns=['column', 'meaning', 'group', 'freq']).set_index('column')
CHARS = dd.index.tolist()
assert set(CHARS) == set(gkx.columns) - {'permno', 'date', 'sic2'}, 'dictionary does not match the file'
dd['dtype'] = gkx[CHARS].dtypes.astype(str)
dd['n_unique'] = [gkx[c].nunique() for c in CHARS]
BINARY = dd.index[dd.n_unique <= 2].tolist()
print('Shape:', gkx.shape, '| dtypes:', gkx.dtypes.value_counts().to_dict())
print('Counts by group:', dd.group.value_counts().to_dict(), '| by update frequency:', dd.freq.value_counts().to_dict())
print('Binary (0/1) columns:', BINARY)

# ---- sample values for every column ----------------------------------------------------------
# integer-valued columns (dummies, scores, codes): list the distinct values with their share of rows;
# continuous columns: show the 1st / 25th / 50th / 75th / 99th percentiles as representative values.
def is_integer_valued(s):
    v = s.dropna()
    return len(v) > 0 and bool(np.all(np.mod(v.values, 1) == 0))
def sample_values(s, max_levels=12):
    v = s.dropna()
    if is_integer_valued(s) and v.nunique() <= max_levels:
        vc = v.value_counts(normalize=True).sort_index()
        return 'levels: ' + ', '.join(f'{k:g} ({100*p:.0f}%)' for k, p in vc.items())
    if is_integer_valued(s):
        return f'integer, {v.nunique()} levels, e.g. ' + ', '.join(f'{x:g}' for x in np.sort(v.unique())[:8]) + ', ...'
    qs = v.quantile([.01, .25, .5, .75, .99]).values
    return 'p1/p25/p50/p75/p99: ' + ' / '.join(f'{x:.3g}' for x in qs)
dd['value_type'] = ['binary 0/1' if c in BINARY else ('integer' if is_integer_valued(gkx[c]) else 'continuous') for c in CHARS]
dd['sample_values'] = [sample_values(gkx[c]) for c in CHARS]
dd['missing_%'] = (100 * gkx[CHARS].isna().mean()).round(2)

# identifiers + industry code, so the saved dictionary covers EVERY column of the file
ids = pd.DataFrame({
    'meaning': ['CRSP permanent security ID (integer key, e.g. 10107 = Microsoft)',
                'Last trading day of the month (row t = information known at end of month t-1, see 2.4)',
                'First 2 digits of the SIC industry code (categorical, e.g. 73 = business services)'],
    'group': ['identifier', 'identifier', 'industry (categorical)'], 'freq': ['-', 'M', 'M'],
    'dtype': [str(gkx[c].dtype) for c in ['permno', 'date', 'sic2']],
    'n_unique': [gkx[c].nunique() for c in ['permno', 'date', 'sic2']],
    'value_type': ['integer ID', 'date', 'categorical code'],
    'sample_values': [', '.join(map(str, gkx.permno.drop_duplicates().sample(5, random_state=0).sort_values())) + ', ...',
                      f"{gkx.date.min():%Y-%m-%d} ... {gkx.date.max():%Y-%m-%d}",
                      'most common: ' + ', '.join(f'{int(k)} ({100*p:.0f}%)' for k, p in gkx.sic2.value_counts(normalize=True).head(6).items())],
    'missing_%': [100 * gkx[c].isna().mean().round(4) for c in ['permno', 'date', 'sic2']]},
    index=pd.Index(['permno', 'date', 'sic2'], name='column'))
data_dict = pd.concat([ids, dd])

DOCS = 'Data Definition and Summary Statistics'          # tables too wide to read in the notebook output
os.makedirs(DOCS, exist_ok=True)
data_dict.to_csv(os.path.join(DOCS, 'gkx_data_dictionary.csv'), encoding='utf-8-sig')
print(f'saved full dictionary ({len(data_dict)} columns) -> {DOCS}/gkx_data_dictionary.csv')
with pd.option_context('display.max_rows', 200, 'display.max_colwidth', 90):
    display(data_dict)

# %% [markdown]
# **Data types.** `permno` is an integer ID (identifier, not a feature). `date` is a month-end date. All 94 characteristics are `float32`:
# * six are 0/1 dummies (`convind, divi, divo, rd, securedind, sin`);
# * four are integer-valued: three small scores (`ps` 0–9, `ms` 0–8, `nincr` 0–8) and `age` (whole years);
# * the rest are continuous.
#
# `sic2` is stored as a float but is really **categorical**: we will one-hot encode it or map it to sectors, never treat it as a number.
#
# **How to read the dictionary columns.**
# * `meaning`: what the characteristic measures.
# * `group`: broad family (accounting, valuation, trading, risk, momentum, firm).
# * `freq`: how often the underlying data are refreshed. **M** = monthly from stock prices; **Q** = quarterly and **A** = annual from company financial statements (Compustat). An annual item keeps the same value for 12 rows in a row.
# * `value_type` and `sample_values`:
#   * For **binary and integer** columns, the actual levels and the share of rows at each level. Example: `ps` is the Piotroski score, an integer from 0 to 9.
#   * For **continuous** columns, five representative values: the 1st, 25th, 50th, 75th and 99th percentiles.
#
# **Files saved.** The full tables are too wide to read in the notebook, so they are also written to `MS2/Data Definition and Summary Statistics/`:
# * `gkx_data_dictionary.csv`: every GKX column with its meaning, group, frequency, dtype, number of distinct values, value type, sample values and % missing.
# * `gkx_summary_statistics.csv`: count, missing %, mean, std, min, 1%, median, 99% and max for every numeric column.
# * `goyal_welch_dictionary_and_stats.csv` and `fama_french5_dictionary_and_stats.csv`: the same for the two market-level files.
#
# ### 2.2 Summary statistics (mean, min, max, …) of every numerical column

# %%
summ = gkx[CHARS + ['sic2']].describe(percentiles=[.01, .5, .99]).T
summ['missing_%'] = 100 * gkx[CHARS + ['sic2']].isna().mean()
summ = summ[['count', 'missing_%', 'mean', 'std', 'min', '1%', '50%', '99%', 'max']]
summ.insert(0, 'meaning', data_dict['meaning'].reindex(summ.index))
summ.to_csv(os.path.join(DOCS, 'gkx_summary_statistics.csv'), encoding='utf-8-sig')
print(f'saved -> {DOCS}/gkx_summary_statistics.csv ({len(summ)} columns)')
with pd.option_context('display.max_rows', 200, 'display.max_colwidth', 40):
    display(summ)

# %% [markdown]
# #### Goyal–Welch monthly macro data (`Data2024.xlsx - Monthly.csv`)
#
# **Plain version.** Each row is **one calendar month** (column `yyyymm`, e.g. 202003 = March 2020). It describes the stock market *as a whole* and the economy: where the S&P 500 stood, how much it paid in dividends and earned, interest rates, how turbulent the market was, and so on. Each value applies to every stock in that month. We use it to tell the model whether the market as a whole is calm or nervous.
#
# **Technical version.** Amit Goyal's 2024 update of the Welch & Goyal (2008) and Goyal, Welch & Zafirov (2024) equity-premium predictor file: 1871-01 → 2024-12, 57 columns. Rates and returns are decimals per month, except bond yields, which are annual decimals.
# * `d/p` = d12/price: for 1871-01, 0.26 / 4.44 = 0.0586, matching the file.
# * Several columns are **empty** in this release (`eqis, cay, i/k, pce, govik, skew, crdstd, accrul, cfacc, gpce, gip, house`). They are quarterly or annual series that are not filled in the monthly sheet, and we drop them.
# * The newer variables are named after their source papers, as cited in GWZ 2024. Check the paper's appendix before relying on the exact construction of any of them.
#
# | Column | Meaning (plain → technical) |
# |---|---|
# | `price` | Level of the S&P 500 index at month end (index points) |
# | `d12` | Dividends paid by S&P 500 firms over the last 12 months (index points) |
# | `e12` | Earnings of S&P 500 firms over the last 12 months (index points) |
# | `ret` / `retx` | S&P 500 return in the month, with / without dividends (CRSP value-weighted) |
# | `AAA` / `BAA` | Yield on Moody's top-rated (Aaa) / lower-investment-grade (Baa) corporate bonds (annual) |
# | `lty` / `ltr` | Long-term U.S. government bond yield / return |
# | `corpr` | Long-term corporate bond return |
# | `tbl` | 3-month Treasury-bill rate (annual) |
# | `Rfree` | Risk-free return for the month (T-bill) |
# | `d/p`, `d/y` | Dividend-price ratio (d12/price); dividend yield (d12/last month's price) |
# | `e/p`, `d/e` | Earnings-price ratio (e12/price); payout ratio (d12/e12) |
# | `b/m` | Book-to-market ratio of the Dow Jones Industrial Average |
# | `tms` | Term spread = `lty` − `tbl` (long minus short rates) |
# | `dfy` | Default yield spread = `BAA` − `AAA` (credit risk premium) |
# | `dfr` | Default return spread = `corpr` − `ltr` |
# | `infl` | CPI inflation (monthly) |
# | `ntis` | Net equity expansion: net share issuance by NYSE firms relative to market value |
# | **`svar`** | **Stock variance: sum of squared *daily* S&P 500 returns in the month = market realized variance. Our main market-turbulence variable.** |
# | `csp` | Cross-sectional beta premium (Polk–Thompson–Vuolteenaho), ends 2002 |
# | `vp`, `vrp` | Variance (risk) premium = implied minus realized variance (Bollerslev–Tauchen–Zhou), 1990+ |
# | `impvar`, `rsvix` | Implied variance from option prices (VIX²-type), and Martin's SVIX, 1996+ |
# | `lzrt` | Log of the number of zero daily returns (market illiquidity) |
# | `ogap` | Output gap (economy above/below trend) |
# | `wtexas` | Change in West Texas Intermediate oil price |
# | `sntm` | Investor sentiment index |
# | `ndrbl` | New orders of durable goods |
# | `skvw`, `tail` | Average stock skewness (value-weighted); tail-risk measure |
# | `fbm` | Cross-sectional book-to-market factor |
# | `dtoy`, `dtoat` | Dow index relative to its 52-week high / all-time high |
# | `ygap` | Yield gap: equity earnings yield vs. bond yield |
# | `rdsp` | Dispersion of stock returns across firms |
# | `tchi` | Technical-indicator index |
# | `avgcor` | Average pairwise correlation between stocks |
# | `shtint` | Aggregate short interest |
# | `disag` | Analyst disagreement |
#
# #### Fama–French 5 factors (`F-F_Research_Data_5_Factors_2x3.csv`)
#
# **Plain version.** Each row is **one calendar month**, 1963-07 → 2026-09. It gives six numbers, each the return in that month of an imaginary "bet" portfolio. Example: "small companies minus big companies" was −0.48% in July 1963, so small firms lagged big firms by about half a percent that month.
#
# **Technical version.** Monthly returns from Kenneth French's library. The file gives them in percent; the loader divides by 100. Each factor except `RF` is a long-short portfolio built from 2×3 sorts of NYSE/AMEX/NASDAQ stocks:
#
# | Column | Meaning |
# |---|---|
# | `Mkt-RF` | Value-weighted U.S. market return minus the 1-month T-bill rate: the market's excess return |
# | `SMB` | Small Minus Big: return of small-cap stocks minus large-cap stocks (size) |
# | `HML` | High Minus Low: return of high book-to-market ("value") minus low ("growth") stocks |
# | `RMW` | Robust Minus Weak: return of high-profitability minus low-profitability firms |
# | `CMA` | Conservative Minus Aggressive: return of low-investment minus high-investment firms |
# | `RF` | 1-month Treasury-bill return (the risk-free rate) |

# %%
GW_DESC = {'price': 'S&P 500 index level', 'd12': '12-month S&P 500 dividends', 'e12': '12-month S&P 500 earnings',
           'ret': 'S&P 500 return incl. dividends', 'retx': 'S&P 500 return excl. dividends', 'AAA': 'Aaa corporate bond yield',
           'BAA': 'Baa corporate bond yield', 'lty': 'long-term govt bond yield', 'ltr': 'long-term govt bond return',
           'corpr': 'long-term corporate bond return', 'tbl': '3-month T-bill rate', 'Rfree': 'risk-free return',
           'd/p': 'dividend-price ratio', 'd/y': 'dividend yield', 'e/p': 'earnings-price ratio', 'd/e': 'payout ratio',
           'b/m': 'DJIA book-to-market', 'tms': 'term spread lty-tbl', 'dfy': 'default yield spread BAA-AAA',
           'dfr': 'default return spread corpr-ltr', 'infl': 'CPI inflation', 'ntis': 'net equity expansion',
           'svar': 'S&P 500 realized variance (sum of squared daily returns)', 'csp': 'cross-sectional beta premium',
           'vp': 'variance premium', 'vrp': 'variance risk premium', 'impvar': 'implied variance', 'rsvix': 'SVIX (Martin)',
           'lzrt': 'log # zero daily returns', 'ogap': 'output gap', 'wtexas': 'WTI oil price change', 'sntm': 'investor sentiment',
           'ndrbl': 'new orders, durable goods', 'skvw': 'avg stock skewness (value-weighted)', 'tail': 'tail risk',
           'fbm': 'cross-sectional book-to-market factor', 'dtoy': 'Dow / 52-week high', 'dtoat': 'Dow / all-time high',
           'ygap': 'yield gap', 'rdsp': 'return dispersion', 'tchi': 'technical-indicator index', 'avgcor': 'average stock correlation',
           'shtint': 'aggregate short interest', 'disag': 'analyst disagreement'}
gw_tab = gw.describe().T[['count', 'mean', 'min', 'max']]
gw_tab.insert(0, 'meaning', pd.Series(GW_DESC).reindex(gw_tab.index).fillna('(empty in this release)'))
gw_tab['first month'] = [str(gw[c].first_valid_index()) for c in gw_tab.index]
gw_tab['last month'] = [str(gw[c].last_valid_index()) for c in gw_tab.index]
gw_tab.to_csv(os.path.join(DOCS, 'goyal_welch_dictionary_and_stats.csv'), encoding='utf-8-sig')
EMPTY_GW = gw_tab.index[gw_tab['count'] == 0].tolist()
print('Goyal-Welch: one row = one calendar month; empty columns (dropped):', EMPTY_GW)
with pd.option_context('display.max_rows', 100): display(gw_tab.drop(EMPTY_GW))

FF_DESC = {'Mkt-RF': 'market excess return', 'SMB': 'small minus big (size)', 'HML': 'high minus low book-to-market (value)',
           'RMW': 'robust minus weak profitability', 'CMA': 'conservative minus aggressive investment', 'RF': '1-month T-bill return'}
ff_tab = ff.describe().T[['count', 'mean', 'std', 'min', 'max']]
ff_tab.insert(0, 'meaning', pd.Series(FF_DESC))
ff_tab.to_csv(os.path.join(DOCS, 'fama_french5_dictionary_and_stats.csv'), encoding='utf-8-sig')
print('Fama-French 5: one row = one calendar month, values = decimal monthly returns. First rows:'); display(ff.head(3)); display(ff_tab)

# %% [markdown]
# ### 2.3 Coverage over time
#
# * **Coverage** here means how many stocks the GKX file has data for in each month. Coverage grows as more companies list and as CRSP adds exchanges: NASDAQ firms enter at the end of 1972. Because rows are lagged one month (§2.4), the jump appears in the row dated 1973-01, where the count goes from about 2,700 to 5,600.
#   *Technically:* `gkx.groupby('date').size()`, the number of `permno` rows per month.
# * **Universe size** is the number of stocks we *keep* for modeling each month. We keep 500, the largest by market value, as a proxy for the S&P 500 (§3.3).
#   The red dashed line shows that every month has far more than 500 stocks available, so the top-500 universe is always full.

# %%
per_month = gkx.groupby('date').size()
fig, ax = plt.subplots(figsize=(10, 3.2))
per_month.plot(ax=ax, color='C0'); ax.axhline(500, ls='--', color='C3', label='500 (our universe size)')
ax.set(title='Number of stocks in the GKX panel per month', xlabel='Month', ylabel='# stocks'); ax.legend()
plt.tight_layout(); plt.show()
print(f'{gkx.permno.nunique():,} distinct stocks; {per_month.size} months; stocks per month: '
      f'min {per_month.min():,}, median {per_month.median():,.0f}, max {per_month.max():,}')

# %% [markdown]
# ### 2.4 Timing check: which month does a row's information describe?
#
# This check decides whether our target leaks. If `retvol` in the row dated *t* were the volatility **of month t**, then using the same row's other columns to predict it would mix in information from inside the month we are forecasting.
#
# **Test.** We compare cross-sectional medians of two GKX columns with outside market data at different shifts:
# * `mom1m` (last month's return) against the Fama–French market return;
# * `retvol` (daily-return volatility) against the Goyal–Welch S&P 500 variance `svar` (sum of squared daily index returns in the month).
#
# **The four variables used in this check:**
#
# | Variable | File → column | Plain meaning | Technical meaning |
# |---|---|---|---|
# | `retvol` | GKX `datashare_small.parquet` (same as `datashare.csv`) → column `retvol` | How jumpy one stock was during a month. A value of 0.02 means the price typically moved about 2% per day. | Standard deviation of the stock's **daily** returns within one month (CRSP daily file). This is the stock's monthly realized volatility, and its log becomes our target. |
# | `mom1m` | GKX → column `mom1m` | How much the stock went up or down over one month. | The stock's 1-month return ("short-term reversal" characteristic). |
# | `svar` | Goyal–Welch `Data2024.xlsx - Monthly.csv` → column `svar` | How turbulent the **whole market** (S&P 500) was during a month. | Sum of squared daily S&P 500 returns in the month: the realized variance of the index. |
# | `Mkt-RF` | Fama–French `F-F_Research_Data_5_Factors_2x3.csv` → column `Mkt-RF` | How much the whole market went up or down in a month, above the safe T-bill rate. | Value-weighted CRSP market return minus the 1-month T-bill return. |
#
# **Why this pairing works.**
# * *Plain:* if the GKX row labelled March really contains February's numbers, then the typical stock's `mom1m` in the March row should move with February's market return, not March's. The same holds for volatility.
# * *Technical:* each month we take the **median across all stocks** of `mom1m` (and of `retvol`). That gives one number per month, which removes stock-specific noise and leaves the market-wide component. We then correlate this monthly series with the outside series shifted by −1, 0 and +1 months. The shift with the highest correlation tells us which month the row describes.

# %%
g = gkx.assign(ym=gkx.date.dt.to_period('M')).groupby('ym').agg(mom1m=('mom1m', 'median'), retvol=('retvol', 'median'))
chk = g.join(ff['Mkt-RF']).join(gw['svar'])
res = []
for k, lab in [(-1, 'month t-1'), (0, 'month t'), (1, 'month t+1')]:
    res.append({'outside data from': lab,
                'corr(median mom1m in row t, market return)': chk.mom1m.corr(chk['Mkt-RF'].shift(-k)),
                'corr(log median retvol in row t, log svar)': np.log(chk.retvol).corr(np.log(chk.svar.shift(-k)))})
timing = pd.DataFrame(res).set_index('outside data from'); display(timing)

# %% [markdown]
# **Result.** `mom1m` in row *t* matches the market return of **month t−1** (correlation ≈ 0.88) and is unrelated to month *t* (≈ 0.06). `retvol` in row *t* tracks the S&P variance of **month t−1** most closely.
#
# So **a row dated t holds information known at the end of month t−1.** This gives a clean, leak-free design:
#
# > **features = row *t* of a stock;  target = `log(retvol)` from row *t+1* of the same stock**, which is the realized volatility of month *t*.
#
# The feature row is fully known before month *t* starts, so no future information enters.

# %% [markdown]
# ## 3. Cleaning & preprocessing of the stock panel  *(owner: YiTong)*
#
# We work in this order:
# 1. Check the panel's integrity.
# 2. Build the target and HAR lags **on the full panel** (so a stock's history before it entered the top 500 still counts).
# 3. Restrict to the investable universe.
# 4. Go through the MS2 checklist: missing data, unusual values, scale, class imbalance, leakage.
#
# Cathy (§5) will finalize the target and HAR features. The versions below are drafts that the EDA needs.
#
# ### 3.1 Integrity
#
# Recall that `permno` is the permanent stock ID and `date` is the last trading day of the month (§2.1). Two checks:
#
# 1. **No duplicates.** Each (`permno`, month) pair should appear only once.
# 2. **Gaps.** We find months that are missing in the middle of a stock's history.
#
# **What `m` and `gap` are, and why we compute them this way.**
#
# *Plain version.* Think of numbering every month since year 0 in a row: Jan 2020 is month 24,241, Feb 2020 is 24,242, and so on. For one stock, we subtract each row's number from the row before it.
# * If the answer is **1**, the two rows are back-to-back months, as they should be.
# * If it is **3**, two months are missing in between: that is a **gap**.
#
# We need this because everything we build looks back or forward in time: "last month's volatility", "the average over the last 12 months", "next month's volatility" (the target). These are only correct if the rows really are consecutive months.
#
# *Technical version.*
# * `m = year*12 + month` maps each date to an integer month index. Consecutive calendar months differ by exactly 1, **including across a year boundary**: Dec 2019 = 2019·12+12 = 24,240 and Jan 2020 = 24,241.
# * We cannot use raw date differences instead, because `date` is the last *trading* day. Months are 28–31 days long and the last trading day moves around (e.g. 2020-02-28 → 2020-03-31 is 32 days), so "30 days" cannot tell consecutive months from a gap.
# * `gkx.groupby('permno')['m'].diff()` takes that difference **within each stock** (rows are sorted by `permno`, `date`), so it never compares two different stocks.
# * `gap` is NaN for a stock's first row, 1 for a normal step, and > 1 when months are missing.
#
# **Why gaps hurt the analysis and the models (negative effects).**
# 1. **Wrong lags.** Without the check, `shift(1)` would call the row from 6 months ago "last month". That would put wrong numbers into the HAR features and blur the persistence we measure.
# 2. **Wrong target.** `shift(-1)` would pair features with a volatility that is *not* next month's. That is a mislabeled response: the forecast horizon would silently change from 1 month to several.
# 3. **Lost rows.** Respecting gaps means a row right after a gap has no `har_3`/`har_12` for 3 or 12 months, and a row right before a gap has no target. We lose those rows, or must impute them.
# 4. **Possible selection bias.** Gaps are often not random: trading halts, exchange moves and data outages happen more for distressed or small firms. Dropping those rows can make the sample look calmer than reality. The cell below measures how large this is.
#
# So all lags and the target are built **within gap-free segments**: `seg` in §3.2 starts a new segment at every gap.

# %%
gkx = gkx.sort_values(['permno', 'date']).reset_index(drop=True)
n_dup = gkx.duplicated(['permno', 'date']).sum()
dates_per_month = gkx.groupby(gkx.date.dt.to_period('M')).date.nunique().max()
gkx['m'] = gkx.date.dt.year * 12 + gkx.date.dt.month                     # integer month index (consecutive months differ by 1)
gap = gkx.groupby('permno')['m'].diff()                                  # months since this stock's previous row (NaN = first row)
print(f'duplicate (permno, date) rows: {n_dup}; max distinct dates per calendar month: {dates_per_month} '
      f'(DATE is the last TRADING day of the month, so we key on the calendar month)')
print(f'rows whose previous row for that stock is NOT the previous month (gaps): {(gap > 1).sum():,} '
      f'({100*(gap > 1).mean():.2f}%)')

# %%
# How big is the gap problem?
g_len = (gap[gap > 1] - 1).astype(int)                                   # number of missing months in each gap
stocks_with_gap = gkx.loc[gap > 1, 'permno'].nunique()
print(f'stocks with at least one gap: {stocks_with_gap:,} of {gkx.permno.nunique():,} '
      f'({100*stocks_with_gap/gkx.permno.nunique():.1f}%)')
print('length of gaps (missing months):'); display(g_len.value_counts().sort_index().head(6).rename('# gaps').to_frame()
        .rename_axis('missing months').T.assign(**{'>6': int((g_len > 6).sum())}))
# a concrete example: one stock's rows around its first gap
ex_pm = gkx.loc[gap > 1, 'permno'].iloc[0]
ex_i = gkx.index[(gkx.permno == ex_pm) & (gap > 1)][0]
ex = gkx.loc[ex_i - 2: ex_i + 1, ['permno', 'date', 'm', 'retvol']].assign(gap=gap.loc[ex_i - 2: ex_i + 1])
print(f'example (permno {ex_pm}): the jump in m shows the missing months'); display(ex)
# is volatility different right before a gap? (selection-bias check)
before_gap = gkx.groupby('permno')['m'].shift(-1).sub(gkx['m']).gt(1)
print(f"median retvol in the month before a gap: {gkx.loc[before_gap, 'retvol'].median():.4f} "
      f"vs all other rows: {gkx.loc[~before_gap, 'retvol'].median():.4f}")

# %% [markdown]
# **What the gap check shows.**
#
# *Plain version.*
# * About 1 stock in 8 has at least one hole in its history. In the example above, permno 10007 jumps from November 1987 to March 1989, so 15 months are missing.
# * Stocks are noticeably **more volatile just before a hole**: a median daily volatility of 3.0% against 2.3% for all other rows. Holes therefore tend to hit the shakier stocks.
#
# *Technical version.*
# * Gaps affect 0.14% of rows but 12.9% of `permno`s. Most are a single missing month, with a long tail of gaps over 6 months (often re-listings).
# * Because pre-gap rows have about 30% higher median `retvol`, dropping them would bias the sample slightly toward calm stock-months.
# * Inside the top-500 universe this matters much less: large stocks rarely have gaps (see the "why is the target missing" table in §3.4).
# * Our handling:
#   1. build lags only inside gap-free segments;
#   2. lose the target only for the single row right before a gap;
#   3. let the size filter, not the gap, decide who is in the sample.

# %% [markdown]
# ### 3.2 Draft target and HAR lags (built on the full panel, respecting gaps)
#
# The realized variance of month *j* is $RV_j = \text{retvol}_j^2$. With row *t* holding month *t−1*:
#
# * target: $y_t = \log \text{retvol}$ of month *t* (taken from row *t+1*; it is kept only if row *t+1* is exactly the next month);
# * `har_1`  $= \log \text{retvol}$ of month *t−1* (row *t*);
# * `har_3`  $= \tfrac12 \log \big(\tfrac13\sum_{j=t-3}^{t-1} RV_j\big)$, i.e. the log volatility of the average variance over the last 3 months;
# * `har_12` is the same over the last 12 months.
#
# Windows never cross a gap in a stock's history. A `retvol` below $10^{-4}$ means a daily std under 0.01%, about 0.16% a year, which is impossible for a traded large stock: it signals a stale price or no trading. Such values are set to missing **before** taking logs (see §3.5).

# %%
STALE = 1e-4                                                          # daily std < 0.01% (~0.16%/yr) = stale price
rv = gkx['retvol'].where(gkx['retvol'] >= STALE)                      # 0 / ~0 -> NaN (see 3.5)
gkx.loc[gkx['baspread'] < 0, 'baspread'] = np.nan                     # impossible negative spread (see 3.5)
gkx['lrv'] = np.log(rv)
nxt_m = gkx.groupby('permno')['m'].shift(-1)
gkx['has_next'] = nxt_m == gkx['m'] + 1
gkx['y'] = gkx.groupby('permno')['lrv'].shift(-1).where(gkx['has_next'])
seg = (gap.fillna(99) > 1).cumsum()                                   # new segment after any gap
var = rv ** 2
for w in (3, 12):
    gkx[f'har_{w}'] = 0.5 * np.log(var.groupby(seg).rolling(w, min_periods=w).mean().reset_index(level=0, drop=True))
gkx['har_1'] = gkx['lrv']
gkx['target_month'] = gkx['date']                                     # y_t is the volatility OF month t
print(gkx[['y', 'har_1', 'har_3', 'har_12']].describe().T[['count', 'mean', 'min', 'max']])

# %% [markdown]
# **Why are the summary statistics of `y` and `har_1` almost identical? This is expected, not an error.**
#
# *Plain version.* `har_1` is "this stock's volatility last month" and `y` is "this stock's volatility this month". Both are drawn from the same long list of monthly volatilities, just one step apart.
#
# Imagine a stock with five months of volatility: 3, 5, 4, 6, 2.
# * The `har_1` column lists the months that come *before* a target month: 3, 5, 4, 6.
# * The `y` column lists the months that come *after* one: 5, 4, 6, 2.
#
# The two columns share every value except one at each end. With millions of rows, losing one value at each end changes the mean, min and max almost not at all. The numbers still sit on **different rows**: on any given row, last month and this month usually differ.
#
# *Technical version.*
# * By construction, $y_{i,t} = \text{har\_1}_{i,t+1}$ whenever row *t+1* is the next month. Both are $\log \text{retvol}$ of the same month, read from two adjacent rows.
# * The marginal distributions of the two columns are therefore the same series shifted by one row. They differ only at:
#   * the first observation of each gap-free segment, which has a `har_1` but is no stock's target;
#   * the last observation, which is a target but not a lag.
# * That is about 1 row per segment out of ~4 M, so the counts differ slightly and the means and extremes coincide. The same holds for any stationary series and its first lag.
# * The real test is **row-wise**, done below:
#   1. $y_{i,t}$ equals the next row's `har_1` exactly;
#   2. the correlation between $y$ and `har_1` *on the same row* is clearly below 1. That correlation is the one-month persistence the models exploit, not a copy of the target.

# %%
nxt_har1 = gkx.groupby('permno')['har_1'].shift(-1).where(gkx['has_next'])
both = gkx['y'].notna() & nxt_har1.notna()
print(f"rows where y_t == har_1 of row t+1 (same month's volatility): {np.isclose(gkx.loc[both, 'y'], nxt_har1[both]).mean()*100:.2f}%")
print(f"count: y {gkx.y.notna().sum():,} vs har_1 {gkx.har_1.notna().sum():,} "
      f"(difference = segment end points and stale/missing retvol)")
pair = gkx[['y', 'har_1']].dropna()
print(f"same-row correlation corr(y_t, har_1_t) = {pair.y.corr(pair.har_1):.3f}; "
      f"mean |y - har_1| = {(pair.y - pair.har_1).abs().mean():.3f} log points "
      f"(~{100*(np.exp((pair.y - pair.har_1).abs().mean())-1):.0f}% typical month-to-month change in volatility)")
ex = gkx.loc[gkx.permno == 10107, ['permno', 'date', 'retvol', 'har_1', 'y']].iloc[300:305]
print('example, Microsoft (permno 10107): each y equals the har_1 of the NEXT row'); display(ex)

# %% [markdown]
# ### 3.3 Universe: the 500 largest stocks each month (S&P 500 proxy)
#
# The proposal targets S&P 500 stocks. GKX covers every CRSP stock, including thousands of micro-caps whose volatility comes from illiquidity rather than information. Each month we keep the **500 largest stocks by lagged market equity `mvel1`**.
#
# Since `mvel1` is lagged, it is known at the time of prediction, so the universe itself does not leak. Stocks that later delisted stay in the months when they were large, which removes the survivorship bias the TF flagged for yfinance.

# %%
gkx['size_rank'] = gkx.groupby('date')['mvel1'].rank(ascending=False, method='first')
uni = gkx[gkx.size_rank <= 500].copy()
tenure = uni.groupby('permno').size()
print(f'universe: {len(uni):,} stock-months, {uni.permno.nunique():,} distinct stocks; '
      f'median months in universe per stock = {tenure.median():.0f}; '
      f'stocks that are in the universe in its last month (2021-12): {uni[uni.date==uni.date.max()].shape[0]}')
print(f'smallest stock admitted, median over months: ${uni.groupby("date").mvel1.min().median()/1e3:,.0f}M (nominal)')
# sector label from SIC divisions (Nic will refine to Fama-French 12 industries in section 7)
def sic_division(s):
    if pd.isna(s): return 'Unknown'
    s = int(s)
    for lo, hi, name in [(1,9,'Agriculture'),(10,14,'Mining/Energy'),(15,17,'Construction'),(20,39,'Manufacturing'),
                         (40,49,'Transport/Utilities/Telecom'),(50,51,'Wholesale'),(52,59,'Retail'),
                         (60,67,'Finance'),(70,89,'Services/Tech'),(90,99,'Public/Other')]:
        if lo <= s <= hi: return name
    return 'Unknown'
uni['sector'] = uni['sic2'].map(sic_division)
display(uni.sector.value_counts(normalize=True).mul(100).round(1).rename('% of stock-months'))

# %% [markdown]
# ### 3.4 Missing data
#
# We look at three things:
# * how much is missing in each column;
# * whether missingness has a **pattern over time** (accounting data starts late in the sample) and **across sectors**;
# * why the **target** is missing. The target is `y`, next month's log `retvol` from GKX row *t+1*.

# %%
miss = uni[CHARS + ['sic2']].isna().mean().mul(100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(15, 4))
colors = dd.reindex(miss.index)['freq'].map({'M': 'C0', 'Q': 'C1', 'A': 'C2'}).fillna('C7')
ax.bar(miss.index, miss.values, color=colors)
ax.set(title='Missing values per column, top-500 universe 1957-2021 (blue = monthly, orange = quarterly, green = annual data)',
       ylabel='% missing', xlabel='column'); ax.tick_params(axis='x', rotation=90, labelsize=7)
plt.tight_layout(); plt.show()
print(f'columns with 0% missing: {(miss == 0).sum()}, with >30% missing: {(miss > 30).sum()}, worst: {miss.index[0]} ({miss.iloc[0]:.0f}%)')

# %%
yr_miss = uni.assign(year=uni.date.dt.year).groupby('year')[CHARS].apply(lambda d: d.isna().mean() * 100)
order = miss.drop('sic2').index
fig, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(yr_miss[order].T, cmap='viridis_r', vmin=0, vmax=100, cbar_kws={'label': '% missing'}, ax=ax)
ax.set(title='Missingness pattern: column x year (top-500 universe)', xlabel='year', ylabel='column')
ax.tick_params(axis='y', labelsize=6); plt.tight_layout(); plt.show()
med_miss = yr_miss.median(axis=1)
first_ok = med_miss[med_miss < 5].index.min()
print('median % missing across columns, selected years:'); print(med_miss.loc[[1960, 1965, 1970, 1975, 1980, 1990, 2000, 2010, 2021]].round(1).to_dict())
print(f'first year in which the median column is <5% missing: {first_ok}')

# %%
modern = uni[uni.date.dt.year >= 1980]
sec_miss = modern.groupby('sector')[CHARS].apply(lambda d: d.isna().mean().mean() * 100).sort_values()
key_cols = ['rd_mve', 'invest', 'cash', 'roaq', 'stdacc', 'bm', 'lev']
sec_tab = modern.groupby('sector')[key_cols].apply(lambda d: d.isna().mean() * 100).round(1)
sec_tab.insert(0, 'avg over all 94', sec_miss)
print('% missing by sector, 1980-2021 (is missingness related to the type of firm?)'); display(sec_tab.sort_values('avg over all 94'))

# %%
tgt = pd.Series({'target available': uni.y.notna().sum(),
                 'no next-month row (delisting, gap, or last month 2021-12)': (~uni.has_next).sum(),
                 'next-month row exists but its retvol is missing or stale (< 1e-4)': (uni.has_next & uni.y.isna()).sum()})
print('Why is the target missing? (top-500 universe)'); display(tgt.to_frame('stock-months'))

# %% [markdown]
# **Missing-data decisions** (to be applied in the modeling pipeline):
# 1. **Modeling window starts in 1980**, with a 1985-start sensitivity check, because 1985 is the first year in which the median column is under 5% missing. Before roughly the mid-1970s most quarterly and many annual Compustat-based columns are almost entirely missing in this universe. That is *missing by design*: the data did not exist yet, so the values are not missing at random. The time-series EDA below uses the full 1957–2021 history of `retvol`, which is never missing.
# 2. **Rows without a target are dropped.**
#    * **What we predict.** The response is `y` = log of `retvol` for month *t*, i.e. the stock's realized volatility in the month we forecast.
#    * **Where it comes from.** It is read from the GKX file (`datashare_small.parquet`, column `retvol`), but from the **next row** (*t+1*) of the same stock (§2.4, §3.2).
#    * **What "without a target" means.** For feature row *t*, `y` is missing when any of these holds:
#      * there is **no row *t+1*** for that `permno`: the stock delisted, was acquired, has a gap, or row *t* is 2021-12, the last month in the file;
#      * row *t+1* exists, but its `retvol` is missing or below the stale-price threshold.
#    * **Why drop rather than impute.** The table above counts each case. We cannot score a forecast without an outcome, and imputing the *response* would make up the answers the model is graded on.
# 3. **Features are imputed with the cross-sectional median of that month** (GKX's approach), each paired with a 0/1 *was-missing* indicator for the columns with heavy missingness.
#    * The median is computed within each month, so imputation never uses future data.
#    * The indicator lets models learn whether missingness itself carries information (MAR/MNAR, Lecture 10).
#    * Columns that are more than 50% missing after 1980 are candidates to drop, and feature selection (lasso) will decide.
# 4. Missingness varies by sector: for example, R&D items are often blank for financial firms. This is another reason for the missing-indicator approach rather than dropping rows.

# %% [markdown]
# ### 3.5 Unusual values

# %%
nonneg = ['mvel1', 'retvol', 'baspread', 'ill', 'turn', 'std_turn', 'std_dolvol', 'idiovol', 'zerotrade', 'age', 'betasq', 'herf']
chk_rows = []
for c in nonneg:
    chk_rows.append((c, 'must be >= 0', int((uni[c] < 0).sum())))
for c in BINARY:
    chk_rows.append((c, 'must be 0/1', int((~uni[c].dropna().isin([0, 1])).sum())))
for c, hi in [('ps', 9), ('ms', 8), ('nincr', 8)]:
    chk_rows.append((c, f'integer score 0-{hi}', int((~uni[c].dropna().between(0, hi)).sum())))
chk_rows.append(('retvol', '< 1e-4 (price essentially never moved)', int((uni.retvol < 1e-4).sum())))
chk_rows.append(('maxret', '> 100% in one day', int((uni.maxret > 1).sum())))
display(pd.DataFrame(chk_rows, columns=['column', 'rule', '# violating stock-months']))

# %%
uni['ann_vol'] = uni.retvol * np.sqrt(252)
top = uni.nlargest(12, 'ann_vol')[['permno', 'date', 'sector', 'ann_vol', 'maxret', 'mvel1']]
top['vol_month'] = (top.date - pd.offsets.MonthEnd(1)).dt.strftime('%Y-%m')
print('Highest annualized volatilities in the universe (vol_month = month the volatility refers to):'); display(top)

# %%
skew = uni.loc[uni.date.dt.year >= 1980, CHARS].skew().abs().sort_values(ascending=False)
q = uni.loc[uni.date.dt.year >= 1980, skew.index[:10]].quantile([0, .01, .5, .99, 1]).T
q.insert(0, '|skew|', skew.iloc[:10])
print('Ten most skewed characteristics (1980+): heavy tails driven by tiny denominators'); display(q)

# %% [markdown]
# **Unusual-value decisions:**
# * **Almost no impossible values.**
#   * The only violation of a hard rule is **one negative bid-ask spread**. A spread cannot be negative, so it is a data error and is **set to missing**.
#   * Everything else (0/1 dummies, bounded scores, non-negative quantities) passes.
# * **`retvol` below $10^{-4}$** (a few dozen stock-months) means the price was essentially unchanged all month: a stale quote or a halt. It is not a real "near-zero risk" reading, and its log would be an extreme outlier (about −20 versus a typical −4). It is **set to missing** in both the target and the lags.
# * **Extreme volatilities are real events** (1987, 2000–02, 2008, 2020, single-stock collapses and bankruptcies), and the stress periods are exactly what the project question asks about. They are **kept**. The log transform tames their influence on squared-error loss.
# * **Extreme characteristics** come from ratios with near-zero denominators (sales/inventory, % changes). They are errors in *scale*, not in fact. Rather than deleting them we **rank-transform each characteristic cross-sectionally to [−1, 1] each month** (GKX). This bounds outliers, removes units, and uses only that month's data, so it does not leak.

# %% [markdown]
# ### 3.6 Scale of the features

# %%
rng = uni.loc[uni.date.dt.year >= 1980, CHARS].agg(['min', 'max']).T
rng['range'] = rng['max'] - rng['min']
rng = rng.sort_values('range')
fig, ax = plt.subplots(figsize=(15, 4))
ax.bar(rng.index, np.log10(rng['range'].clip(lower=1e-6)), color='C0')
ax.set(title='Range (max - min) of each characteristic, log10 scale, 1980-2021', ylabel='log10(range)', xlabel='column')
ax.tick_params(axis='x', rotation=90, labelsize=7); plt.tight_layout(); plt.show()
print(f"ranges span {np.log10(rng['range'].min()):.1f} to {np.log10(rng['range'].max()):.1f} orders of magnitude "
      f"(smallest: {rng.index[0]}, largest: {rng.index[-1]})")

def cs_rank(df, cols):                        # cross-sectional rank -> [-1, 1], NaN preserved
    r = df.groupby('date')[cols].rank(pct=True)
    return 2 * r - 1
demo = uni.loc[uni.date.dt.year >= 1980, ['date', 'mvel1', 'bm', 'saleinv']]
demo_r = cs_rank(demo, ['mvel1', 'bm', 'saleinv'])
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
demo[['mvel1', 'bm', 'saleinv']].describe().loc[['min', '50%', 'max']].T.plot.barh(ax=axes[0], logx=True)
axes[0].set(title='Raw scale (log axis)', xlabel='value')
demo_r.describe().loc[['min', '50%', 'max']].T.plot.barh(ax=axes[1])
axes[1].set(title='After monthly rank transform to [-1, 1]', xlabel='value'); plt.tight_layout(); plt.show()

# %% [markdown]
# **Scale decision.** The characteristics differ by many orders of magnitude: `mvel1` is in thousands of dollars, while most ratios are near 0. Ridge, lasso, kNN and neural networks all need comparable scales.
# * **Stock characteristics:** monthly rank transform to [−1, 1], as above.
# * **HAR lags and macro/factor variables:** z-scored with the mean and standard deviation of the *training window only*, refit at every walk-forward step to avoid leakage.
#
# Trees do not need scaling, but using the same inputs for all models keeps the comparison fair.
#
# ### 3.7 Class imbalance → *imbalance in regimes, not classes*
# This is a regression task, so there are no classes. The analogous problem is that **stress months are rare**, while the project asks whether gains survive in stress periods. We define stress months as months whose S&P realized variance (`svar`) is in the top 10% of the *training-period* distribution, and count them.

# %%
svar_m = gw['svar'].copy()
thr = svar_m.loc['1980-01':'2010-12'].quantile(0.90)                    # threshold from training period only
stress = (svar_m > thr).rename('stress')
s = stress.loc['1980-01':'2021-11']
by = pd.DataFrame({'stress months': [s.loc[:'2010-12'].sum(), s.loc['2011-01':'2015-12'].sum(), s.loc['2016-01':].sum()],
                   'total months': [s.loc[:'2010-12'].size, s.loc['2011-01':'2015-12'].size, s.loc['2016-01':].size]},
                  index=['train 1980-2010', 'validation 2011-2015', 'test 2016-2021'])
by['% stress'] = 100 * by['stress months'] / by['total months']; display(by)
print('stress months in test window:', [str(p) for p in s.loc['2016-01':][s.loc['2016-01':]].index])

# %% [markdown]
# The test window contains only a handful of stress months, which are concentrated in the 2020 COVID shock, with 2018 Q4 borderline. A "stress-period" comparison on the test set alone therefore rests on few independent episodes. The plan has two parts:
# 1. Report the calm/stress split on the test set **with bootstrap confidence intervals**, resampling whole months (block bootstrap).
# 2. Add a *secondary* out-of-sample check over the 2008–09 crisis, where models are trained only on data up to 2007.
#
# ### 3.8 Information-leakage audit (stock panel)

# %%
X = uni.loc[uni.y.notna() & (uni.date.dt.year >= 1980)]
corr_y = X[CHARS + ['har_1', 'har_3', 'har_12']].corrwith(X['y']).abs().sort_values(ascending=False)
print('Largest |correlation| between a feature in row t and the target y_t (month t volatility):')
display(corr_y.head(10).to_frame('|corr with y|'))
print('Any feature with |corr| > 0.95 (would suggest the feature IS the target)?', bool((corr_y > 0.95).any()))

# %% [markdown]
# **Leakage checks done (stock panel):**
# 1. **Timing verified empirically (§2.4).** Row *t* = information up to the end of month *t−1*, and the target comes strictly from row *t+1*.
# 2. **No feature is a disguised copy of the target.** The strongest correlations are with past volatility (`retvol`/HAR lags, `idiovol`, `maxret`, `baspread`), roughly 0.7–0.8. That is expected persistence, not leakage, because they all describe month *t−1* or earlier.
# 3. **The universe is chosen with lagged size**, so we never select on future information. Delisted stocks remain in the sample.
# 4. **All preprocessing statistics are computed either within a month (ranks, medians) or on the training window only (z-scores, stress threshold).**
# 5. **A 1-month embargo** separates each training window from the next validation/test window. Monthly targets do not overlap, so this is a safety margin rather than a necessity. (With overlapping weekly targets the original plan needed a 5-day gap.)
#
# Anthony (§6) repeats this audit on the merged table that includes macro data, which have their own publication lags.
#
# ### 3.9 Save the cleaned stock panel

# %%
keep = ['permno', 'date', 'target_month', 'sic2', 'sector', 'size_rank', 'y', 'har_1', 'har_3', 'har_12'] + CHARS
clean = uni.loc[uni.y.notna(), keep].reset_index(drop=True)
clean.to_parquet(os.path.join(OUT, 'panel_top500_clean.parquet'), index=False)
modeling = clean[clean.date.dt.year >= 1980]
print(f'cleaned panel (all years): {clean.shape[0]:,} rows x {clean.shape[1]} cols; '
      f'modeling window 1980-2021: {modeling.shape[0]:,} rows, {modeling.permno.nunique():,} stocks; '
      f'file = {os.path.getsize(os.path.join(OUT, "panel_top500_clean.parquet"))/2**20:.0f} MB')

# %% [markdown]
# ## 4. Time-series EDA  *(owner: YiTong)*
#
# MS3 is where the full EDA happens. Here we look at the **time-series properties of the response**, because they determine which models make sense. Anything that feeds a modeling choice (ARIMA order, HAR clustering) uses **target months up to 2015 only**, so the test period stays untouched.
#
# ### 4.1 Distribution of the response

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
v = uni.loc[uni.y.notna()]
axes[0].hist(np.exp(v.y) * np.sqrt(252), bins=150, color='C0'); axes[0].set_xlim(0, 1.5)
axes[0].set(title='Monthly realized volatility (annualized)', xlabel='annualized volatility', ylabel='stock-months')
axes[1].hist(v.y, bins=150, color='C1')
axes[1].set(title='log of daily-return std (our target y)', xlabel='y = log(retvol)', ylabel='stock-months')
plt.tight_layout(); plt.show()
print(f'raw vol skewness {pd.Series(np.exp(v.y)).skew():.2f}  vs  log vol skewness {v.y.skew():.2f}')

# %% [markdown]
# Raw volatility is strongly right-skewed, while its log is close to symmetric. That is the standard reason to model **log** volatility: squared-error loss is not dominated by a few crash months, and linear models fit far better.
#
# ### 4.2 Volatility through time: clustering and regimes

# %%
agg = uni.groupby('date')['lrv'].agg(med='median', q10=lambda s: s.quantile(.1), q90=lambda s: s.quantile(.9))
vol_month = agg.index - pd.offsets.MonthEnd(1)            # row t describes month t-1
mkt_vol = np.sqrt(gw['svar'] * 12)
fig, ax = plt.subplots(figsize=(13, 4.2))
ax.fill_between(vol_month, np.exp(agg.q10) * np.sqrt(252), np.exp(agg.q90) * np.sqrt(252), color='C0', alpha=.25, label='10th-90th percentile across stocks')
ax.plot(vol_month, np.exp(agg.med) * np.sqrt(252), color='C0', lw=1.2, label='median stock')
ax.plot(mkt_vol.loc['1957-01':'2021-12'].index.to_timestamp(), mkt_vol.loc['1957-01':'2021-12'].values, color='C3', lw=.9, label='S&P 500 index (Goyal-Welch svar)')
for a, b, lab in STRESS_EPISODES:
    ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color='grey', alpha=.2)
    ax.text(pd.Timestamp(a), 1.07, lab, fontsize=8, rotation=90, va='top')
ax.set(title='Monthly realized volatility, top-500 stocks, 1957-2021 (grey = stress episodes)', xlabel='month', ylabel='annualized volatility', ylim=(0, 1.1))
ax.legend(loc='upper left'); plt.tight_layout(); plt.show()

# %% [markdown]
# Three features are visible:
# 1. **Volatility clusters**: calm and turbulent months come in runs.
# 2. Every stress episode raises *all* stocks together, so a large share of each stock's volatility is a **common market factor**. This is why market-level inputs (lagged `svar`, VIX-like variables) should help a pooled model.
# 3. Single-stock volatility sits well above index volatility because diversification cancels idiosyncratic risk.
#
# ### 4.3 Persistence: autocorrelation of log volatility

# %%
agg_ts = pd.Series(agg['med'].values, index=pd.PeriodIndex(vol_month, freq='M'))
fit_ts = agg_ts.loc[:'2015-12']
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
L = 36
a, ci = acf(fit_ts, nlags=L, alpha=.05); p, pci = pacf(fit_ts, nlags=L, alpha=.05)
axes[0].bar(range(1, L + 1), a[1:]); axes[0].fill_between(range(1, L + 1), ci[1:, 0] - a[1:], ci[1:, 1] - a[1:], color='grey', alpha=.3)
axes[0].set(title='ACF: median log vol (1957-2015)', xlabel='lag (months)', ylabel='autocorrelation')
axes[1].bar(range(1, L + 1), p[1:]); axes[1].fill_between(range(1, L + 1), pci[1:, 0] - p[1:], pci[1:, 1] - p[1:], color='grey', alpha=.3)
axes[1].set(title='PACF: median log vol', xlabel='lag (months)', ylabel='partial autocorrelation')
# pooled within-stock ACF: demean each stock, then correlate x_t with x_{t-k} only across consecutive months
P = gkx.loc[gkx.permno.isin(uni.permno.unique()) & (gkx.date <= '2016-01-31'), ['permno', 'm', 'lrv']].dropna()
P['x'] = P.lrv - P.groupby('permno').lrv.transform('mean')
base = P.set_index(['permno', 'm'])['x']
pooled = []
for k in range(1, L + 1):
    lagged = P.assign(m=P.m + k).set_index(['permno', 'm'])['x']
    j = pd.concat([base, lagged.rename('xl')], axis=1, join='inner')
    pooled.append(np.corrcoef(j.iloc[:, 0], j.xl)[0, 1])
axes[2].plot(range(1, L + 1), pooled, 'o-', ms=3)
for k in (1, 3, 12): axes[2].axvline(k, color='C3', ls=':', lw=1)
axes[2].set(title='Pooled within-stock ACF (red: HAR lags 1/3/12)', xlabel='lag (months)', ylabel='autocorrelation', ylim=(0, 1))
plt.tight_layout(); plt.show()
print('pooled within-stock ACF at lags 1, 3, 6, 12, 24, 36:', np.round([pooled[i - 1] for i in (1, 3, 6, 12, 24, 36)], 3))
lb = acorr_ljungbox(fit_ts, lags=[6, 12, 24], return_df=True); display(lb)


# %% [markdown]
# The ACF **decays slowly**: correlations are still clearly positive after two to three years. This is *long memory*.
# * A single AR(1) term would make the ACF decay geometrically, which is far too fast to match this.
# * The **HAR model** handles it cheaply by averaging past volatility over a short horizon (1 month), a medium one (3 months) and a long one (12 months).
#
# The partial autocorrelations beyond the first few lags are small. The Ljung–Box test strongly rejects "no autocorrelation". So volatility *is* forecastable, which is the premise of the project.
#
# ### 4.4 Stationarity

# %%
def stat_tests(x):
    adf_p = adfuller(x, autolag='AIC')[1]; kpss_p = kpss(x, regression='c', nlags='auto')[1]
    return adf_p, kpss_p
adf_p, kpss_p = stat_tests(fit_ts.values)
print(f'median log vol 1957-2015: ADF p = {adf_p:.4f} (H0: unit root)  |  KPSS p = {kpss_p:.3f} (H0: stationary)')
# per-stock: stocks with >= 120 consecutive months of lrv before 2016
P2 = gkx.loc[gkx.permno.isin(uni.permno.unique()) & (gkx.date <= '2016-01-31'), ['permno', 'lrv']].dropna()
cnt = P2.groupby('permno').size(); elig = cnt[cnt >= 120].index
pick = RNG.choice(elig, size=min(300, len(elig)), replace=False)
out = []
for pm in pick:
    x = P2.loc[P2.permno == pm, 'lrv'].values
    out.append(stat_tests(x))
out = np.array(out)
print(f'{len(pick)} random stocks with >=120 months: ADF rejects unit root (p<.05) for {100*(out[:,0]<.05).mean():.0f}%; '
      f'KPSS rejects stationarity (p<.05) for {100*(out[:,1]<.05).mean():.0f}%')

# %% [markdown]
# **How to read the two tests.**
# * **ADF** tests the null "unit root", meaning shocks never die out; rejecting it is evidence *for* stationarity.
# * **KPSS** tests the null "stationary"; rejecting it is evidence *against* stationarity.
#
# When ADF rejects *and* KPSS also rejects, that is the classic signature of a **long-memory (fractionally integrated) series**. It is mean-reverting, but too slowly for a short-memory model.
#
# **Implications:**
# * Model **levels of log volatility** without first-differencing them, since differencing would throw away the mean-reversion that makes forecasting possible.
# * Use HAR-type averages to capture the slow decay.
# * Expect slow drifts in the volatility level over decades, so use **expanding or rolling training windows** and re-estimate the models over time.
#
# ### 4.5 Automatic ARIMA order selection (analogue of R's `auto.arima`)
# `pmdarima` is not available, so we run a small AIC grid search with `statsmodels`. This is the same logic `auto.arima` uses.

# %%
y_fit = fit_ts.loc['1980-01':].astype(float).values
grid = []
for d in (0, 1):
    for p_ in range(0, 4):
        for q_ in range(0, 4):
            try:
                r = ARIMA(y_fit, order=(p_, d, q_), trend='c' if d == 0 else 'n').fit()
                grid.append({'order': (p_, d, q_), 'AIC': r.aic, 'BIC': r.bic})
            except Exception:
                pass
grid = pd.DataFrame(grid).sort_values('AIC')
display(grid.head(6))
best = ARIMA(y_fit, order=grid.iloc[0]['order'], trend='c' if grid.iloc[0]['order'][1] == 0 else 'n').fit()
print('best by AIC:', grid.iloc[0]['order'], '| Ljung-Box p of residuals (lag 12):',
      round(acorr_ljungbox(best.resid, lags=[12], return_df=True)['lb_pvalue'].iloc[0], 3))

# %% [markdown]
# The AIC grid chooses **ARMA(3,3) on levels (d = 0)**, and its residuals pass the Ljung–Box test. Two points support this:
# * The undifferenced model winning agrees with the ADF result that the series is mean-reverting.
# * Needing several AR *and* MA terms is another symptom of long memory: a short-memory model has to stack lags to mimic slow decay. HAR does the same job with only 3 coefficients.
#
# For the market-wide series this gives a second "traditional time-series" baseline. ARIMA is also the first stage of the optional *ARIMA + ML-on-residuals* hybrid described in the outline.
#
# ### 4.6 Per-stock HAR fits and clustering stocks by their volatility dynamics
# For each stock with at least 120 usable months whose target month is before 2016, we fit
# $y_t = b_0 + b_1\,\text{har\_1} + b_2\,\text{har\_3} + b_3\,\text{har\_12} + \varepsilon_t$ by OLS.

# %%
H = gkx.loc[gkx.permno.isin(uni.permno.unique()) & (gkx.date <= VAL_END), ['permno', 'y', 'har_1', 'har_3', 'har_12']].dropna()
cnt = H.groupby('permno').size(); H = H[H.permno.isin(cnt[cnt >= 120].index)]
coefs = []
for pm, g in H.groupby('permno'):
    Xh = np.column_stack([np.ones(len(g)), g[['har_1', 'har_3', 'har_12']].values]); yh = g.y.values
    b, *_ = np.linalg.lstsq(Xh, yh, rcond=None)
    r2 = 1 - ((yh - Xh @ b) ** 2).sum() / ((yh - yh.mean()) ** 2).sum()
    coefs.append((pm, *b, r2, len(g)))
C = pd.DataFrame(coefs, columns=['permno', 'b0', 'b_1m', 'b_3m', 'b_12m', 'R2', 'n']).set_index('permno')
C['persistence'] = C[['b_1m', 'b_3m', 'b_12m']].sum(axis=1)
print(f'{len(C)} stocks; median in-sample R2 = {C.R2.median():.2f}'); display(C.describe().T[['mean', '25%', '50%', '75%']])
fig, axes = plt.subplots(1, 4, figsize=(15, 3))
for ax, c in zip(axes, ['b_1m', 'b_3m', 'b_12m', 'persistence']):
    ax.hist(C[c].clip(C[c].quantile(.01), C[c].quantile(.99)), bins=40); ax.set(title=f'per-stock {c}', xlabel='coefficient', ylabel='# stocks')
plt.tight_layout(); plt.show()

# %% [markdown]
# **What the table `C` is.** One row per stock (index = `permno`) holding that stock's own fitted HAR model:
#
# | Column | Meaning |
# |---|---|
# | `b0` | intercept: the stock's baseline log-volatility level |
# | `b_1m`, `b_3m`, `b_12m` | weights on last month's (`har_1`), last quarter's (`har_3`) and last year's (`har_12`) volatility |
# | `R2` | share of the variation in that stock's monthly log volatility the fit explains, in-sample |
# | `n` | number of months used for the fit |
# | `persistence` | `b_1m + b_3m + b_12m`: how much of a volatility shock carries into next month. Below 1 means it fades over time (mean reversion). |
#
# So `C` turns ~2,100 long time series into ~2,100 short "fingerprints" of how each stock's volatility remembers the past.
#
# **What the clustering does: grouping stocks with similar volatility memory.**
#
# *Plain version.* Picture each stock as a dot on a 3-D chart, placed by its three weights (1-month, 3-month, 12-month). Some stocks react mostly to last month, others mostly to last year. Clustering asks the computer to sort the dots into **k groups** so that dots in the same group sit close together. **k is the number of groups**; k = 2 means "two types of stock". Each stock gets a group label, saved in the column `C['cluster']`.
#
# *Technical version.*
# * **Inputs.** The three weights are clipped at their 1st/99th percentiles (a handful of short-history stocks have extreme estimates). They are then standardized (`StandardScaler`: mean 0, std 1), so that no weight dominates the distances just because of its scale.
# * **K-means** (`KMeans(k)`) picks k cluster centres to minimise the total squared distance from each stock to its nearest centre. It alternates two steps: assign each stock to its nearest centre, then move each centre to the mean of its stocks. `n_init=10` restarts from 10 random starts and keeps the best; `random_state=0` makes the result reproducible.
# * **Choosing k with the silhouette score** (`silhouette_score`). For each stock, let *a* = its mean distance to the other stocks in its own cluster and *b* = its mean distance to the stocks of the nearest *other* cluster. Its silhouette is $(b-a)/\max(a,b)$.
#   * The value is in [−1, 1]. **1** means clearly in the right group; **0** means on the border between two groups; **negative** means probably in the wrong group.
#   * The score is the average over stocks.
#   * We try k = 2, …, 6 and keep the k with the highest average. That is an automatic, data-driven choice of k, in the same spirit as choosing ARIMA orders by AIC.
# * **Sector check.** We cross-tabulate the clusters against sector, to see whether they just rediscover industries.

# %%
# Z: stocks x 3 matrix of the HAR weights, clipped at 1%/99% and standardized -> the points to cluster
Z = StandardScaler().fit_transform(C[['b_1m', 'b_3m', 'b_12m']].clip(C[['b_1m', 'b_3m', 'b_12m']].quantile(.01), C[['b_1m', 'b_3m', 'b_12m']].quantile(.99), axis=1))
# for each candidate number of groups k: run k-means, then score how well-separated the groups are (silhouette, -1..1)
sil = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=0).fit_predict(Z)) for k in range(2, 7)}
k_best = max(sil, key=sil.get)                                       # k with the best separation
C['cluster'] = KMeans(k_best, n_init=10, random_state=0).fit_predict(Z)
sec_of = uni.groupby('permno').sector.agg(lambda s: s.mode().iloc[0])
C['sector'] = sec_of.reindex(C.index)
print('silhouette by k:', {k: round(v, 3) for k, v in sil.items()}, '-> k =', k_best)
display(C.groupby('cluster')[['b_1m', 'b_3m', 'b_12m', 'persistence', 'R2']].median().assign(n_stocks=C.cluster.value_counts()))
display(pd.crosstab(C.sector, C.cluster, normalize='columns').mul(100).round(0))

# %% [markdown]
# **Reading the HAR fits.**
# * For the median stock, all three horizons get positive weight (about 0.21 / 0.31 / 0.26), with a persistence (their sum) of about 0.80 < 1, i.e. mean reversion. The median in-sample R² is about 0.39.
# * The spread of coefficients across stocks shows that **stocks differ in how their volatility remembers shocks**.
# * The silhouette score picks **k = 2** clusters:
#   * a **long-memory** group, where the 12-month weight dominates;
#   * a **medium-memory** group, where the 3-month weight dominates.
# * Both clusters have similar sector mixes, so the dynamics are **not just an industry effect**. The silhouette scores are modest (about 0.3), so this is a soft structure rather than sharply separated groups.
#
# Clustering is an *optional extension* under the staff caveats. Its possible use in MS3/MS4 is to add `cluster` as a feature, or to fit one pooled model per cluster (partial pooling), which sits between per-stock HAR and one pooled model for all stocks.
#
# ### 4.7 Cross-section: size and sector

# %%
m80 = uni[(uni.date.dt.year >= 1980) & uni.y.notna()].copy()
m80['size_decile'] = m80.groupby('date').mvel1.transform(lambda s: pd.qcut(s.rank(method='first'), 10, labels=False) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.boxplot(data=m80.sample(60000, random_state=0), x='size_decile', y='y', ax=axes[0], showfliers=False, color='C0')
axes[0].set(title='Next-month log vol by size decile (1 = smallest of top 500)', xlabel='size decile within universe', ylabel='y = log(retvol)')
order_s = m80.groupby('sector').y.median().sort_values().index
sns.boxplot(data=m80.sample(60000, random_state=0), y='sector', x='y', order=order_s, ax=axes[1], showfliers=False, color='C1')
axes[1].set(title='Next-month log vol by sector', xlabel='y = log(retvol)', ylabel='')
plt.tight_layout(); plt.show()

# %% [markdown]
# Inside the top 500, volatility falls *modestly* with size. The largest decile is the calmest, and the effect is much weaker than across all CRSP stocks. Volatility also differs systematically across sectors: utilities/telecom and finance are the calmest, services/tech and mining/energy the most volatile. 'Unknown' (missing `sic2`, about 5% of rows) is the most dispersed group. These are **cross-sectional** signals that a per-stock HAR cannot use but a pooled model can. They are the main hypothesis for why ML could beat HAR.
#
# ### 4.8 Seasonality

# %%
m80['dev'] = m80.y - m80.har_12                                   # vol relative to the stock's own 12-month level
m80['dev'] -= m80['dev'].mean()                                    # remove constant Jensen gap (log of mean var > mean of log)
m80['cal_month'] = m80.target_month.dt.month
sea = m80.groupby('cal_month').dev.agg(['mean', 'sem'])
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(sea.index, sea['mean'], yerr=1.96 * sea['sem'], color='C2', capsize=3)
ax.set(title='Calendar-month effect: log vol vs own 12-month level, demeaned (1980-2021)', xlabel='calendar month of the target', ylabel='mean deviation (log)')
ax.set_xticks(range(1, 13)); plt.tight_layout(); plt.show()

# %% [markdown]
# There is a real but small calendar pattern:
# * **October** is the most volatile month (1987, 2008 and other autumn sell-offs);
# * **December** and late spring/summer are the calmest;
# * January, April, July and October are earnings-season months, but the pattern is not purely an earnings-season effect.
#
# The effect is small compared with swings across regimes. It costs almost nothing to include as month-of-year dummies, so we will let the models decide whether it is worth keeping.
#
# #### Should we use a seasonal GARCH model?
#
# *Plain version.* A GARCH model is a classic tool that guesses "how nervous will the market be tomorrow" from how big recent price moves were. A *seasonal* (periodic) GARCH also lets the answer depend on the time of year, e.g. "Octobers tend to be rougher".
#
# It is a sensible idea, but it fits our data poorly for three reasons:
# 1. **We already observe volatility directly.** GARCH was built for the case where you only see returns and must *infer* volatility from them. We are given each month's measured volatility (`retvol`), which is more accurate.
# 2. **GARCH needs many returns per stock.** With monthly data a stock has about 12 returns a year. A seasonal GARCH would need to estimate several extra parameters for each calendar month from roughly 30–40 Octobers per stock, which is far too little data.
# 3. **The calendar effect is small** (see the chart above).
#
# So we test the *same idea* in the simpler form that suits our data: add 11 calendar-month on/off variables to the HAR regression and check whether forecasts improve.
#
# *Technical version.*
# * **Where seasonality would enter.** A periodic GARCH, e.g. Bollerslev & Ghysels (1996), $\sigma^2_t = \omega_{s(t)} + \alpha_{s(t)}\varepsilon^2_{t-1} + \beta_{s(t)}\sigma^2_{t-1}$ with season $s(t)$, models the *latent* conditional variance of *returns*.
# * **Why a realized-volatility model is better here.** Realized volatility is a far less noisy measure of $\sigma_t$ than a squared monthly return, so HAR-on-RV models generally forecast better than GARCH on the same horizon. GARCH would also need per-stock monthly return series, which we only have indirectly through `mom1m`, and $\omega_s, \alpha_s, \beta_s$ would be poorly identified with about 40 observations per season.
# * **The test: HAR with calendar dummies.** The seasonal analogue in our framework is a HAR-X model, $y_t = b_0 + \sum_{m=2}^{12} \gamma_m \mathbb{1}[\text{month}(t)=m] + b_1\text{har\_1} + b_2\text{har\_3} + b_3\text{har\_12} + \varepsilon_t$.
#   * It is fit pooled on target months 1980–2010 and scored on the 2011–2015 validation months. The test window is never touched.
# * **Optional extension for MS3.** If this helps, fit a GARCH(1,1) with calendar dummies on a *market* return series (`ff['Mkt-RF']`, about 750 months) as an extra traditional baseline for the aggregate series. `arch` supports regressors only in the mean equation, so seasonal variance terms would need a custom likelihood. That is a reasonable item for Cathy's §5 if time allows.

# %%
# Pooled HAR vs HAR + calendar-month dummies: train target months 1980-2010, evaluate on validation 2011-2015
S = uni.loc[(uni.date.dt.year >= 1980) & (uni.target_month <= VAL_END), ['y', 'har_1', 'har_3', 'har_12', 'target_month']].dropna()
S['cal_month'] = S.target_month.dt.month
def design(d, seasonal):
    X = [np.ones(len(d)), d.har_1, d.har_3, d.har_12]
    if seasonal:
        X += [(d.cal_month == mth).astype(float) for mth in range(2, 13)]       # January = baseline
    return np.column_stack(X)
tr, va = S[S.target_month <= TRAIN_END], S[S.target_month > TRAIN_END]
rows = []
for name, seas in [('HAR', False), ('HAR + 11 calendar-month dummies', True)]:
    Xtr, Xva = design(tr, seas), design(va, seas)
    b, *_ = np.linalg.lstsq(Xtr, tr.y.values, rcond=None)
    rss = ((tr.y.values - Xtr @ b) ** 2).sum()
    rows.append({'model': name, 'n params': Xtr.shape[1], 'train RSS': rss,
                 'train MSE': rss / len(tr), 'validation MSE': ((va.y.values - Xva @ b) ** 2).mean()})
res_s = pd.DataFrame(rows).set_index('model')
F = ((res_s['train RSS'].iloc[0] - res_s['train RSS'].iloc[1]) / 11) / (res_s['train RSS'].iloc[1] / (len(tr) - res_s['n params'].iloc[1]))
res_s['validation MSE change vs HAR (%)'] = 100 * (res_s['validation MSE'] / res_s['validation MSE'].iloc[0] - 1)
display(res_s.drop(columns='train RSS'))
print(f'train rows {len(tr):,}, validation rows {len(va):,}; nested F-statistic for the 11 dummies (train) = {F:.1f}. '
      f'Caveat: rows are not independent (stocks in the same month move together), so this F overstates '
      f'significance; validation MSE is the honest check.')

# %% [markdown]
# **Result of the seasonality test.**
#
# *Plain version.* Telling the model which calendar month it is forecasting makes its validation-period errors only **about 1% smaller**. That is a real but tiny gain. A seasonal GARCH would spend far more effort to capture the same small effect, so we skip it and keep the month labels as a cheap optional input.
#
# *Technical version.*
# * **In-sample:** the nested F-statistic is very large (~540, with 11 and ~180k degrees of freedom). It is inflated, though, because errors are strongly correlated within a month: 500 stocks share one market shock. With only ~31 Octobers in training, the effective sample for each calendar month is small.
# * **Out-of-sample (2011–2015):** validation MSE falls by about 0.9% (0.1253 → 0.1242). That validation window has only 5 observations of each calendar month, so the gain is fragile.
# * **Decision:**
#   * month-of-year dummies are a candidate feature for the pooled models, kept only if they survive walk-forward validation and lasso selection;
#   * no seasonal GARCH;
#   * a market-level GARCH with calendar dummies is listed as an optional traditional baseline.
#
# ### 4.9 Market-state link (preview of Anthony's merge)

# %%
mk = pd.DataFrame({'med_y': uni[uni.y.notna()].groupby(uni.target_month.dt.to_period('M')).y.median()})
lag1 = gw.shift(1)                                                 # value known at the end of month t-1
for c in ['svar', 'tbl', 'dfy', 'tms', 'd/p', 'ntis']:
    mk[c + ' (t-1)'] = lag1[c]
mk['|Mkt-RF| (t-1)'] = ff['Mkt-RF'].abs().shift(1)
mk['log svar (t-1)'] = np.log(mk.pop('svar (t-1)'))
mk = mk.loc['1980-01':'2015-12']
print('Correlation of the median next-month log vol with lagged market variables (1980-2015):')
display(mk.corr()['med_y'].drop('med_y').sort_values(key=abs, ascending=False).to_frame('corr'))

# %% [markdown]
# ### 4.10 What the time-series EDA tells us to do
#
# | Finding | Evidence | Consequence for modeling |
# |---|---|---|
# | Log vol is near-symmetric, raw vol is skewed | §4.1 | target = log volatility; evaluate with MSE on logs **and** QLIKE on variances |
# | Strong, slowly decaying autocorrelation (long memory) | §4.3–4.4 | HAR (1/3/12-month) baseline; keep levels; expanding or rolling re-estimation |
# | Common market component in every stress episode | §4.2, §4.9 | add lagged market variance, credit spread, and factor returns as pooled features |
# | ARMA(3,3) chosen by AIC for the aggregate series (several AR and MA terms = long-memory symptom) | §4.5 | ARIMA baseline for market vol; residual-hybrid as an optional extension |
# | Stocks differ in HAR dynamics; soft clusters | §4.6 | per-stock HAR as the main benchmark; cluster label or partial pooling as an option |
# | Size and sector shift the volatility level | §4.7 | cross-sectional features are the main hypothesized source of ML gains |
# | Stress months are rare in the test window | §3.7 | block-bootstrap CIs; secondary 2008–09 out-of-sample check |
# | Small calendar effect (October high, December low) | §4.8 | month-of-year dummies in pooled models (HAR-X), not a seasonal GARCH; keep only if validation loss improves |

# %% [markdown]
# ---
# ## 5. Target & volatility features  *(owner: **Cathy**)*
#
# **Input:** the full GKX panel `gkx` and the top-500 universe `uni` from §3 (still in memory), plus the Goyal–Welch market variance `svar`.
# **Output:** `data_clean/panel_top500_features.parquet` (same rows as `panel_top500_clean.parquet` plus the new columns), and the baseline functions `predict_baseline`, `qlike`, `to_variance` and `dm_test` for MS3.
#
# | § | Step | Result |
# |---|---|---|
# | 5.1 | One function builds the target and every volatility feature | `build_vol_features()`; reproduces the §3.2 drafts exactly |
# | 5.2 | Leakage test by truncation | every model input is unchanged when the future is deleted |
# | 5.3 | Final target and partial months | target definition, partial-month flag, summary statistics |
# | 5.4 | HAR features and the 12-month warm-up | strict and "filled" HAR lags, rows lost to warm-up |
# | 5.5 | Extra volatility features, EWMA λ | EWMA, vol-of-vol, stock-vs-market vol; λ chosen on validation |
# | 5.6 | Unusual jumps in the target | >5× month-to-month moves, classified and kept |
# | 5.7 | Baselines for MS3 | EWMA, pooled HAR, per-stock HAR with a cold-start rule; QLIKE with bias correction |
# | 5.8 | Save | the feature table and the numbers for the revised proposal |

# %% [markdown]
# ### 5.1 One builder for the target and all volatility features
#
# *Plain version.* All volatility features are built by **one function**. We run it once on the whole panel, and again in §5.2 on a copy with the future deleted. If both runs give the same numbers, the features cannot be peeking at the future.
#
# *Technical version.* Row *t* of GKX holds information about month *t−1* (§2.4). $RV_j = \text{retvol}_j^2$ is the realized variance of month *j*. Every window is computed **within a gap-free run** of a stock (`seg`, as in §3.2), so a lag never jumps over missing months.
#
# | Column | Definition | Latest month of information | Model input? |
# |---|---|---|---|
# | `y` | $\log \text{retvol}$ of month *t* (read from row *t+1*, kept only if it is the next month) | **t** (the answer) | target |
# | `har_1` | $\log \text{retvol}$ of month *t−1* | t−1 | yes |
# | `har_3`, `har_12` | $\tfrac12\log$ of the mean $RV$ over the last 3 / 12 months; the **full** window is required (= §3.2) | t−1 | yes |
# | `har_3_f`, `har_12_f` | the same, but using whatever history exists (at least 1 month) | t−1 | yes |
# | `har_3_n`, `har_12_n` | number of months actually inside the window (≤ 3 / ≤ 12) | t−1 | yes |
# | `ewma_10` … `ewma_90` | $\tfrac12\log$ of an EWMA of $RV$: $s = \lambda s_{prev} + (1-\lambda) RV$, with λ = 0.1, 0.2, 0.3, 0.5, 0.7, 0.9 per month | t−1 | yes |
# | `volvol_12` | standard deviation of $\log \text{retvol}$ over the last 12 months (at least 6) | t−1 | yes |
# | `mkt_lvol` | market daily log vol of month *t−1*: $\tfrac12\log(\text{svar}/21)$ | t−1 | yes (overlaps with Anthony's `svar`; keep one) |
# | `rel_vol` | `har_1 − mkt_lvol`: is the stock calmer or jumpier than the market? | t−1 | yes |
# | `flag_first` | row is the first of a run (listing or relisting), so `har_1` may come from a partial month | t−1 | yes |
# | `flag_tgt_last` | the target comes from the run's last row (just before a delisting or a gap) | **future** | **no**, evaluation only |
# | `partial_month` | `flag_first` or `flag_tgt_last` | **future** | **no**, evaluation only |
#
# **Why λ is not 0.94.** RiskMetrics' 0.94 is a *daily* decay. Over a 21-day month it compounds to $0.94^{21} \approx 0.27$, so a monthly EWMA needs a much smaller λ than 0.94. We try a grid and pick λ on the validation years (§5.5).
#
# **Why `flag_tgt_last` is not a feature.** Knowing that row *t+1* is a stock's last row means knowing that it delists, which is future information. It is kept only to check whether results change without those rows.

# %%
# =============================================================================================
# 5.1  ONE builder for the target and every volatility feature.
#      Used (a) on the full panel and (b) in the leakage test of 5.2, so the test checks
#      exactly the code that produces the modeling table.
# =============================================================================================
import statsmodels.api as sm

LAMS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)   # EWMA decay per MONTH (0.94 is a DAILY value: 0.94**21 ~ 0.27/month)
DAYS_PER_MONTH = 21                # converts svar (sum of daily squared returns in a month) to a daily variance
EWMA_COLS = [f'ewma_{int(round(l * 100))}' for l in LAMS]

def build_vol_features(p, svar, lams=LAMS, stale=STALE):
    """
    p    : panel with columns permno, date, retvol (one row per stock-month; row t = info of month t-1).
    svar : Goyal-Welch S&P 500 realized variance, monthly PeriodIndex, value = variance OF that month.
    Returns a DataFrame aligned to p.index. Feature columns use rows <= t only;
    `y`, `flag_tgt_last`, `partial_month` look forward by design and are never model inputs.
    """
    p = p[['permno', 'date', 'retvol']].sort_values(['permno', 'date'])   # sorting keeps the original index
    m = p['date'].dt.year * 12 + p['date'].dt.month                       # integer month: consecutive = +1
    stock = p['permno']
    seg = (m.groupby(stock).diff().fillna(99) > 1).cumsum()               # id of each gap-free run of a stock
    rv = p['retvol'].where(p['retvol'] >= stale)                          # stale / zero vol -> NaN (§3.5)
    var = rv ** 2                                                         # realized variance of month t-1
    f = pd.DataFrame(index=p.index)

    # ---- target: log vol of month t = row t+1, only if row t+1 is exactly the next month ----
    f['lrv'] = np.log(rv)
    has_next = m.groupby(stock).shift(-1).eq(m + 1)
    f['y'] = f['lrv'].groupby(stock).shift(-1).where(has_next)

    # ---- HAR lags: average the VARIANCES in the window, then 1/2 log (= log of the average vol level) ----
    f['har_1'] = f['lrv']
    for w in (3, 12):
        roll = var.groupby(seg).rolling(w, min_periods=1)                 # window never crosses a gap
        mean = roll.mean().reset_index(level=0, drop=True)
        cnt = roll.count().reset_index(level=0, drop=True)                # months with a valid RV in the window
        f[f'har_{w}_f'] = 0.5 * np.log(mean)                              # "filled": uses the history that exists
        f[f'har_{w}_n'] = cnt
        f[f'har_{w}'] = f[f'har_{w}_f'].where(cnt == w)                   # strict: full window (= §3.2 draft)

    # ---- EWMA (RiskMetrics) of variance; the recursion restarts after every gap ----
    for lam, col in zip(lams, EWMA_COLS):
        e = var.groupby(seg).ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()
        f[col] = 0.5 * np.log(e.reset_index(level=0, drop=True))

    # ---- vol-of-vol: how unstable the stock's own volatility was over the last 12 months ----
    f['volvol_12'] = (f['lrv'].groupby(seg).rolling(12, min_periods=6).std()
                      .reset_index(level=0, drop=True))

    # ---- stock vol relative to market vol, both describing month t-1 ----
    vol_month = p['date'].dt.to_period('M') - 1                           # row t describes month t-1 (§2.4)
    mkt = 0.5 * np.log(svar / DAYS_PER_MONTH)                             # market daily log vol, per month
    f['mkt_lvol'] = vol_month.map(mkt).astype(float)
    f['rel_vol'] = f['har_1'] - f['mkt_lvol']

    # ---- partial-month flags ----
    run_first = seg.ne(seg.shift(1))                                      # first row of a run (listing / relisting)
    run_last = seg.ne(seg.shift(-1))                                      # last row of a run (delisting / gap)
    f['flag_first'] = run_first                                           # har_1 may be a partial month: known at t
    f['flag_tgt_last'] = has_next & run_last.shift(-1, fill_value=True)   # y from the run's last row: FUTURE info
    f['partial_month'] = f['flag_first'] | f['flag_tgt_last']
    return f

FEATURES = ['har_1', 'har_3', 'har_12', 'har_3_f', 'har_12_f', 'har_3_n', 'har_12_n',
            *EWMA_COLS, 'volvol_12', 'mkt_lvol', 'rel_vol', 'flag_first']      # allowed model inputs
LOOK_AHEAD = ['y', 'flag_tgt_last', 'partial_month']                          # labels / metadata only

# %%
t0 = time.time()
F = build_vol_features(gkx, gw['svar'])
print(f'built {F.shape[1]} columns for {len(F):,} stock-months in {time.time() - t0:.0f}s')

def same_values(a, b):
    """True if two columns have NaN in the same rows and (numerically) equal values elsewhere."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])

# the strict versions must reproduce YiTong's §3.2 drafts exactly: same rows, same values
for c in ['y', 'har_1', 'har_3', 'har_12']:
    ok = same_values(F[c], gkx[c])
    print(f'{c:7s} identical to the §3.2 draft: {ok}')
    assert ok, f'{c} differs from §3.2'

NEW = [c for c in F.columns if c not in ('lrv', 'y', 'har_1', 'har_3', 'har_12')]
gkx[NEW] = F[NEW].reindex(gkx.index)
uni[NEW] = F[NEW].reindex(uni.index)      # uni keeps gkx's row index (§3.3), so rows line up
print('new columns:', NEW)

# %% [markdown]
# ### 5.2 Leakage test by truncation
#
# *Plain version.* If a feature only looks at the past, deleting everything after some date cannot change its values *before* that date. We delete all rows after June 2005 for 300 random stocks, rebuild the features, and compare.
#
# *Technical version.* For a random set of universe stocks we run `build_vol_features` on (a) their full history and (b) only rows dated ≤ 2005-06-30. On the rows that exist in both, every column in `FEATURES` must be identical. The look-ahead columns (`y`, `flag_tgt_last`, `partial_month`) **should** change at the cutoff, because they read the next row. Seeing them change shows the test is able to detect a look-ahead.

# %%
CUT = pd.Timestamp('2005-06-30')
test_ids = RNG.choice(uni.permno.unique(), size=min(300, uni.permno.nunique()), replace=False)
P = gkx.loc[gkx.permno.isin(test_ids), ['permno', 'date', 'retvol']]
full = build_vol_features(P, gw['svar'])                        # sees the whole history
trunc = build_vol_features(P[P.date <= CUT], gw['svar'])        # the future after CUT is deleted
leak = pd.Series({c: same_values(full.loc[trunc.index, c], trunc[c]) for c in FEATURES + LOOK_AHEAD},
                 name='unchanged after deleting the future')
display(leak.to_frame().T)
assert leak[FEATURES].all(), 'a model input changed when future rows were removed -> leakage'
print('All model inputs pass. Look-ahead columns that changed (as they should):',
      [c for c in LOOK_AHEAD if not leak[c]])

# %% [markdown]
# ### 5.3 Final target and partial months
#
# **Target (final).** $y_t = \log \text{retvol}$ of month *t*: the log standard deviation of the stock's daily returns in the month we forecast. It is read from GKX row *t+1* of the same stock, and only if that row is exactly the next month. `retvol` below $10^{-4}$ (stale price) is set to missing before taking logs (§3.5).
#
# **Partial months.** The milestone plan asked us to treat months with fewer than about 15 trading days separately (IPO and delisting months). **GKX has no count of trading days**, so this cannot be checked directly. We use the closest proxy available, the edges of each gap-free run:
# * `flag_first`: the first row of a run, so `har_1` may describe a listing month with only a few trading days;
# * `flag_tgt_last`: the target comes from the last row before a delisting or gap.
#
# The cell below compares flagged and unflagged stock-months in the modeling window.

# %%
U = uni[(uni.date.dt.year >= 1980) & uni.y.notna()]                 # modeling rows: 1980+, target observed
lab = U.partial_month.map({False: 'full months', True: 'possibly partial'})
partial_tab = U.groupby(lab).agg(stock_months=('y', 'size'), median_y=('y', 'median'))
partial_tab['share %'] = 100 * partial_tab.stock_months / len(U)
partial_tab['median |y - har_1|'] = (U.y - U.har_1).abs().groupby(lab).median()
partial_tab['flag_first'] = U.groupby(lab).flag_first.sum()
partial_tab['flag_tgt_last'] = U.groupby(lab).flag_tgt_last.sum()
display(partial_tab)

fig, ax = plt.subplots(figsize=(8, 3.2))
for k, g in U.groupby(lab):
    ax.hist(g.y, bins=120, density=True, histtype='step', lw=1.5, label=f'{k} (n={len(g):,})')
ax.set(title='Target y = log realized vol: full vs possibly partial months (1980-2021)',
       xlabel='y = log(retvol)', ylabel='density'); ax.legend(); plt.tight_layout(); plt.show()

# final target summary, in logs and as annualized volatility (daily std x sqrt(252))
qs = [.01, .05, .25, .5, .75, .95, .99]
target_summary = pd.DataFrame({'y (log daily vol)': U.y.describe(percentiles=qs),
                               'annualized vol': np.exp(U.y).mul(np.sqrt(252)).describe(percentiles=qs)})
target_summary.loc['count'] = len(U)
display(target_summary.T)

# %% [markdown]
# **Decision on partial months (rule fixed before looking at test data).**
# * **Keep** the flagged rows. They are only **0.65%** of the modeling rows (1,621 of 250,719). Their median volatility is about the same as other rows (−4.15 vs −4.14), but their month-to-month change is larger (median |y − har_1| 0.31 vs 0.25). They are harder to forecast, not calmer, so dropping them would flatter every model.
# * `flag_first` is known at forecast time, so the models **may use it as an input**.
# * `flag_tgt_last` and `partial_month` use future information. They are **not inputs**. In MS3 we report the main test-set loss with and without these rows as a robustness check.

# %% [markdown]
# ### 5.4 HAR features and the 12-month warm-up
#
# *Plain version.* `har_12` needs 12 months of history. A stock that just listed, or just came back after a gap, has no `har_12` for its first year. The strict version would drop those stock-months.
#
# *Technical version.* We count how many modeling rows lose each strict HAR lag, and whether those rows differ from the rest. The "filled" versions `har_3_f` / `har_12_f` average over the months that exist, and `har_3_n` / `har_12_n` record how many there were. Together they let a model tell a 2-month average from a 12-month one.

# %%
warm = pd.DataFrame({'missing in modeling rows (%)': U[['har_1', 'har_3', 'har_12', 'har_3_f', 'har_12_f']].isna().mean() * 100})
warm['rows lost'] = U[warm.index].isna().sum()
display(warm.round(2))
short = U.har_12_n < 12
print(f'stock-months with < 12 months of history: {short.sum():,} ({short.mean():.2%}); '
      f'median y {U.y[short].median():.3f} vs {U.y[~short].median():.3f} for the rest')
print(f'stocks affected: {U.loc[short, "permno"].nunique():,} of {U.permno.nunique():,}')

# %% [markdown]
# **Decision on the warm-up.**
# * **Models use `har_1`, `har_3_f`, `har_12_f`, plus `har_12_n`.** Only 35 modeling rows have no HAR forecast at all. The strict `har_12` would drop **4,601 rows (1.84%) from 597 of 2,458 stocks**, and those rows are clearly **more volatile** (median y −3.91 vs −4.15, about 27% higher volatility). Dropping them would bias the sample toward calm, established stocks.
# * The strict `har_3` / `har_12` stay in the table, so the textbook HAR (Corsi 2009) can be reported on the rows where it is defined.
# * `har_1` is missing only where last month's `retvol` was stale (§3.5). Those few rows get no HAR forecast and are left out of every model comparison, so all models are scored on the same rows.

# %% [markdown]
# ### 5.5 Extra volatility features and the EWMA λ
#
# The new features join the volatility-related GKX characteristics already in the table (`maxret`, `idiovol`, `baspread`, `ill`). All of them describe month *t−1* or earlier.
# * **Range, missing share, and rank correlation with the target** (Spearman: `ill` and `volvol_12` are heavily skewed, so ranks are more meaningful than raw values).
# * **Choosing λ.** For each λ we score the EWMA on the validation years in two ways:
#   * *raw*: the forecast is exactly the EWMA, as in RiskMetrics, with nothing estimated;
#   * *calibrated*: $y = a + b\cdot\text{ewma}$, fit on the training years.
#   We keep the λ with the lowest calibrated validation MSE. The test years are not used. If the best λ is at the edge of the grid, extend the grid.

# %%
VOL_X = EWMA_COLS + ['volvol_12', 'mkt_lvol', 'rel_vol', 'maxret', 'idiovol', 'baspread', 'ill']
feat_tab = U[VOL_X].describe().T[['mean', 'min', 'max']]
feat_tab.insert(0, 'missing %', U[VOL_X].isna().mean() * 100)
feat_tab['Spearman corr with y'] = U[VOL_X].corrwith(U.y, method='spearman')
display(feat_tab.round(4))

tr = U[U.target_month <= TRAIN_END]
va = U[(U.target_month > TRAIN_END) & (U.target_month <= VAL_END)]
lam_rows = []
for lam, c in zip(LAMS, EWMA_COLS):
    a, b = tr[[c, 'y']].dropna(), va[[c, 'y']].dropna()
    coef = np.polyfit(a[c], a.y, 1)                                       # y = coef[0]*ewma + coef[1]
    lam_rows.append({'lambda': lam, 'raw validation MSE': ((b.y - b[c]) ** 2).mean(),
                     'calibrated validation MSE': ((b.y - np.polyval(coef, b[c])) ** 2).mean(),
                     'slope b': coef[0]})
lam_tab = pd.DataFrame(lam_rows).set_index('lambda'); display(lam_tab)
LAM_STAR = lam_tab['calibrated validation MSE'].idxmin()
EWMA_STAR = EWMA_COLS[LAMS.index(LAM_STAR)]
print(f'chosen lambda = {LAM_STAR} -> column {EWMA_STAR} (selected on validation {TRAIN_END.year + 1}-{VAL_END.year} only)')

fig, ax = plt.subplots(figsize=(6.5, 3))
lam_tab[['raw validation MSE', 'calibrated validation MSE']].plot(ax=ax, marker='o')
ax.set(title='EWMA: validation MSE by decay λ (monthly)', xlabel='λ per month', ylabel='MSE (log vol)')
plt.tight_layout(); plt.show()

# %% [markdown]
# **Reading the table.**
# * All volatility features are positively rank-correlated with next month's volatility (EWMA ≈ 0.68–0.71, `baspread` 0.69, `maxret` / `idiovol` ≈ 0.55, `rel_vol` 0.36, `mkt_lvol` 0.34). That is the persistence seen in §4.3, seen through different windows. `volvol_12` (0.07) and `ill` (0.10) carry little information on their own; lasso can decide whether they help in combination.
# * **λ = 0.5 per month** has the lowest calibrated validation MSE (0.1365). The curve is flat between 0.3 and 0.7 and much worse at 0.9, so the choice is not on the edge of the grid and is not fragile.
# * The gap between *raw* and *calibrated* EWMA error has two sources. Taking the log of an average variance overstates the average log vol (Jensen's inequality), and a pure EWMA has no pull back toward the long-run level. A calibrated or HAR-type model fixes both.
# * `mkt_lvol` carries the same information as Goyal–Welch `svar` (log and rescaled), which Anthony merges in §6. **Keep only one in the final feature list** so that lasso does not split weight between two copies.

# %% [markdown]
# ### 5.6 Unusual jumps in the target
#
# *Plain version.* We look for stock-months where volatility is **more than 5 times** last month's, or **less than one fifth** of it. Either could be a data error or a real event (a crash, a takeover bid, a bankruptcy). We cannot see daily prices, so we classify each jump by its context.
#
# *Technical version.* Jump: $|y_t - \text{har\_1}_t| > \log 5$. Each jump gets the first matching label:
# 1. **market-wide stress month**: the target month is a stress month (S&P `svar` above the training 90th percentile, §3.7);
# 2. **listing/delisting edge**: `partial_month` is set;
# 3. **one-month spike that reverts**: next month's volatility is back within 50% of the pre-jump level. This is the pattern of a one-off event, and also the pattern a data error would leave;
# 4. **persistent shift**: the new level persists (a company event such as a merger, restructuring or distress).

# %%
y_next = gkx['y'].groupby(gkx.permno).shift(-1).where(gkx.has_next)      # vol of month t+1 (to test reversal)
J = U.assign(log_ratio=U.y - U.har_1, y_next=y_next.reindex(U.index))
J = J[J.log_ratio.abs() > np.log(5)].copy()
J['direction'] = np.where(J.log_ratio > 0, 'up (>5x)', 'down (<1/5)')
J['stress'] = J.target_month.dt.to_period('M').map(stress).fillna(False).astype(bool)
J['reverts'] = (J.y_next - J.har_1).abs() < np.log(1.5)
J['class'] = np.select([J.stress, J.partial_month, J.reverts],
                       ['market-wide stress month', 'listing/delisting edge', 'one-month spike that reverts'],
                       default='persistent shift (company event)')
jump_tab = pd.crosstab(J['class'], J['direction'], margins=True, margins_name='total')
display(jump_tab)
print(f'jumps: {len(J):,} of {len(U):,} modeling rows ({len(J) / len(U):.3%})')

top = J.reindex(J.log_ratio.abs().sort_values(ascending=False).index).head(15)
display(pd.DataFrame({'permno': top.permno, 'vol month': top.target_month.dt.strftime('%Y-%m'), 'sector': top.sector,
                      'prev ann. vol': np.exp(top.har_1) * np.sqrt(252), 'ann. vol': np.exp(top.y) * np.sqrt(252),
                      'ratio': np.exp(top.log_ratio), 'class': top['class']}).round(3))

ax = pd.crosstab(J.target_month.dt.year, J['class']).plot.bar(stacked=True, figsize=(13, 3.4), width=.85)
ax.set(title='Target jumps (>5x or <1/5 of last month) per year, by context', xlabel='year of the target month',
       ylabel='# stock-months'); ax.legend(fontsize=8); plt.tight_layout(); plt.show()

uni['jump5'] = (uni.y - uni.har_1).abs() > np.log(5)                    # uses y -> metadata, never an input

# %% [markdown]
# **Decision on jumps.**
# * Jumps are rare: **586 of 250,719 modeling rows (0.23%)**. Upward jumps (300) are mostly **market-wide stress months** (178, e.g. 2008 and 2020). Downward jumps (286) are mostly **persistent shifts** (216).
# * **The largest downward jumps have a recognizable pattern.** Volatility drops from 70–200% a year to 1–4% a year and stays there. That is what happens to the target of a **pending cash acquisition**: once a deal is announced, the price is pinned near the offer price until the deal closes, and the stock is then delisted. We cannot confirm each case without deal data, but it explains why most downward jumps are "persistent" and why some sit at a delisting edge. These are real prices, not errors.
# * **Keep all of them.** §3.5 already ruled out impossible values. A merger-pinned stock is a real, very calm stock-month that no HAR model can foresee from its own past.
# * The log target already limits how much one jump can move squared-error loss.
# * `jump5` is saved as a flag (look-ahead, **not an input**). In MS3 we check that the model ranking does not change when these rows are dropped, or when a Huber loss replaces squared error.

# %% [markdown]
# ### 5.7 Baselines for MS3, and fair scoring
#
# **Baselines.**
# * **EWMA** (RiskMetrics): the forecast is `ewma_λ*`, with nothing estimated.
# * **Pooled HAR**: one OLS of $y$ on (`har_1`, `har_3_f`, `har_12_f`) over all universe stock-months in the training years.
# * **Per-stock HAR**: one OLS per stock. It is fit on the stock's whole GKX history in the training years, including months before it entered the top 500, which gives more data per stock.
#   * **Cold-start rule:** a stock with fewer than `min_hist = 36` training months falls back to pooled HAR. The median stock spends only 57 months in the universe (§3.3), so this rule decides how many stocks get their own model. It is fixed here, before any model comparison.
# * **GARCH(1,1) is dropped.** It needs a stock's return series at a usable frequency, and GKX gives one monthly return (`mom1m`). A "GARCH recursion on `retvol²`" would just be an autoregression of realized variance, which HAR and EWMA already cover.
#
# **Scoring.**
# * **MSE** on log volatility.
# * **QLIKE** on variance, $\text{QLIKE} = RV/h - \log(RV/h) - 1$, where $h$ is the forecast variance. QLIKE ranks models correctly even with a noisy volatility proxy (Patton 2011).
# * **Bias correction.** A model of *log* vol must be turned into a variance forecast first. If $\log \text{vol} \sim N(\hat y, s^2)$, then $E[\text{var}] = \exp(2\hat y + 2s^2)$. Using $\exp(2\hat y)$ alone gives a forecast that is systematically too low, so every log model would look worse on QLIKE than it is. `to_variance` adds the correction, with $s^2$ the residual variance from training.
# * **Diebold–Mariano.** Average the loss difference across stocks within each month first, because stocks in the same month share one market shock. Then test the monthly series with Newey–West standard errors.

# %%
HAR_X = ['har_1', 'har_3_f', 'har_12_f']        # filled lags: every stock with >= 1 month of history gets a forecast

def qlike(y_log, h):
    """QLIKE loss per row. y_log = realized log vol, h = forecast VARIANCE. 0 = perfect; robust to a noisy proxy."""
    r = np.exp(2 * np.asarray(y_log, float)) / np.asarray(h, float)
    return r - np.log(r) - 1

def to_variance(yhat_log, s2):
    """Log-vol forecast -> unbiased variance forecast: log vol ~ N(yhat, s2) => E[vol^2] = exp(2*yhat + 2*s2)."""
    return np.exp(2 * np.asarray(yhat_log, float) + 2 * np.asarray(s2, float))

def _ols(X, y):
    """OLS with intercept; returns (coefficients, residual variance)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    e = y - X1 @ b
    return b, e @ e / max(len(y) - X1.shape[1], 1)

def predict_baseline(train, test, method, hist=None, min_hist=36, ewma_col=None):
    """
    train    : universe rows of the training years (need y + HAR_X); used for the pooled fit.
    test     : rows to forecast (need permno + HAR_X, or the EWMA column).
    method   : 'ewma' | 'har_pooled' | 'har_stock'.
    hist     : rows for the per-stock fits (default = train); pass the full GKX panel to use pre-universe months.
    min_hist : per-stock HAR needs >= min_hist training months, otherwise the stock falls back to pooled HAR.
    Returns DataFrame(index=test.index) with yhat (log vol), s2 (residual var), h (variance forecast), source.
    """
    out = pd.DataFrame({'yhat': np.nan, 's2': np.nan, 'source': None}, index=test.index)
    if method == 'ewma':                                    # RiskMetrics: nothing is estimated
        col = ewma_col or EWMA_STAR
        out['yhat'], out['s2'], out['source'] = test[col], 0.0, col
        out['h'] = np.exp(2 * out.yhat)                     # the EWMA itself is a variance forecast
        return out
    fit = train.dropna(subset=HAR_X + ['y'])
    b, s2 = _ols(fit[HAR_X].to_numpy(float), fit.y.to_numpy(float))
    ok = test[HAR_X].notna().all(axis=1)
    out.loc[ok, 'yhat'] = np.column_stack([np.ones(ok.sum()), test.loc[ok, HAR_X].to_numpy(float)]) @ b
    out.loc[ok, ['s2', 'source']] = s2, 'pooled'
    if method == 'har_stock':
        H = (train if hist is None else hist).dropna(subset=HAR_X + ['y'])
        test_rows = test[ok].groupby('permno').groups       # permno -> test row labels
        for pm, g in H[H.permno.isin(test_rows.keys())].groupby('permno'):
            if len(g) < min_hist:
                continue                                    # cold start: keep the pooled forecast
            bs, s2s = _ols(g[HAR_X].to_numpy(float), g.y.to_numpy(float))
            idx = test_rows[pm]
            out.loc[idx, 'yhat'] = np.column_stack([np.ones(len(idx)), test.loc[idx, HAR_X].to_numpy(float)]) @ bs
            out.loc[idx, ['s2', 'source']] = s2s, 'own'
    out['h'] = to_variance(out.yhat, out.s2)
    return out

def dm_test(loss_a, loss_b, month):
    """Diebold-Mariano on d_t = monthly cross-sectional mean of (loss_a - loss_b), Newey-West SEs.
    Returns (mean d, t-stat, two-sided p). mean d > 0 means model b has the lower loss."""
    d = pd.Series(np.asarray(loss_a, float) - np.asarray(loss_b, float)).groupby(np.asarray(month)).mean()
    lags = max(1, int(4 * (len(d) / 100) ** (2 / 9)))
    r = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return d.mean(), r.tvalues[0], r.pvalues[0]


# %% [markdown]
# **Smoke test on the validation years.** This only checks that the baselines run and gives a first idea of their accuracy. It is not the model comparison, which uses the untouched test years in MS3. All models are scored on the same rows, the ones where every baseline has a forecast.

# %%
hist = gkx[(gkx.target_month.dt.year >= 1980) & (gkx.target_month <= TRAIN_END) & gkx.permno.isin(va.permno.unique())]
preds = {m: predict_baseline(tr, va, m, hist=hist) for m in ['ewma', 'har_pooled', 'har_stock']}
common = pd.concat([p.yhat.notna() for p in preds.values()], axis=1).all(axis=1)
common = common[common].index
yv, month_v = va.loc[common, 'y'], va.loc[common, 'target_month']

rows, losses = [], {}
for m, p in preds.items():
    se = (yv - p.loc[common, 'yhat']) ** 2
    losses[m] = se
    rows.append({'baseline': m, 'coverage %': 100 * p.yhat.notna().mean(), 'MSE (log vol)': se.mean(),
                 'QLIKE': qlike(yv, p.loc[common, 'h']).mean(),
                 'own-model share %': 100 * (p.loc[common, 'source'] == 'own').mean() if m == 'har_stock' else np.nan})
nb = preds['har_pooled']                                     # same forecast, WITHOUT the bias correction
rows.append({'baseline': 'har_pooled, no bias correction', 'MSE (log vol)': losses['har_pooled'].mean(),
             'QLIKE': qlike(yv, np.exp(2 * nb.loc[common, 'yhat'])).mean()})
base_tab = pd.DataFrame(rows).set_index('baseline')
print(f'validation target months {TRAIN_END.year + 1}-{VAL_END.year}: {len(common):,} stock-months scored')
display(base_tab.round(4))

for a, b in [('har_pooled', 'har_stock'), ('ewma', 'har_pooled')]:
    dbar, t, pv = dm_test(losses[a], losses[b], month_v)
    print(f'DM (MSE) {a} vs {b}: mean loss diff = {dbar:.4f}, t = {t:.2f}, p = {pv:.3f}')

# how sensitive is per-stock HAR to the cold-start rule? (choice stays at 36; reported for transparency)
sens = {}
for mh in (12, 36, 60):
    p = predict_baseline(tr, va, 'har_stock', hist=hist, min_hist=mh)
    sens[mh] = {'own-model share %': 100 * (p.loc[common, 'source'] == 'own').mean(),
                'MSE (log vol)': ((yv - p.loc[common, 'yhat']) ** 2).mean()}
display(pd.DataFrame(sens).T.rename_axis('min_hist').round(4))

# %% [markdown]
# **Reading the smoke test.**
# * The **no-bias-correction** row uses the same pooled-HAR forecast, so its MSE is identical, but its QLIKE is worse. This is why every log model is scored through `to_variance` in MS3.
# * **The cold-start rule changes how hard the benchmark is to beat.** A lower `min_hist` gives more stocks their own noisy model; a higher one gives more stocks the pooled model. We keep 36, chosen before the comparison, and report the sensitivity.
# * **Pooled HAR beats per-stock HAR** on validation (MSE 0.1256 vs 0.1306; DM p = 0.013). Even with `min_hist = 60`, per-stock HAR is worse (0.1290), so this is not just the cold-start rule. With ~30–60 months per stock, three free coefficients per stock are estimated too noisily, and pooling the slopes wins. **This matters for the project question:** beating per-stock HAR alone is a low bar. The pooled ML models should be judged against **pooled HAR**, the stronger benchmark.
# * EWMA is clearly worst (MSE 0.1473; DM p < 0.001 against pooled HAR): it has no mean reversion and no intercept.
# * The bias correction matters: pooled-HAR QLIKE improves from 0.300 to 0.274 with the same forecasts.

# %% [markdown]
# ### 5.8 Save the feature table and the numbers for the revised proposal

# %%
NEW_X = ['har_3_f', 'har_12_f', 'har_3_n', 'har_12_n', *EWMA_COLS, 'volvol_12', 'mkt_lvol', 'rel_vol', 'flag_first']
META = ['flag_tgt_last', 'partial_month', 'jump5']          # look-ahead: evaluation / robustness only
feat = uni.loc[uni.y.notna(), keep + NEW_X + META].reset_index(drop=True)
assert len(feat) == len(clean) and feat[['permno', 'date']].equals(clean[['permno', 'date']]), 'rows changed'
path = os.path.join(OUT, 'panel_top500_features.parquet')
feat.to_parquet(path, index=False)
print(f'saved {path}: {feat.shape[0]:,} rows x {feat.shape[1]} cols, {os.path.getsize(path) / 2**20:.0f} MB '
      f'(same rows as panel_top500_clean; {len(NEW_X)} new inputs + {len(META)} look-ahead flags)')

numbers = {
    'feature table (all years)': f'{feat.shape[0]:,} rows x {feat.shape[1]} cols',
    'modeling rows 1980-2021': f'{len(U):,}',
    'possibly partial months': f"{100 * U.partial_month.mean():.2f}% of modeling rows (kept, flagged)",
    'rows without a strict har_12': f"{100 * U.har_12.isna().mean():.2f}% (kept via har_12_f)",
    'EWMA lambda (validation)': LAM_STAR,
    'target jumps >5x or <1/5': f'{len(J):,} ({100 * len(J) / len(U):.2f}%; kept, flagged)',
    'per-stock HAR with own model (validation)': f"{base_tab.loc['har_stock', 'own-model share %']:.1f}%",
    'validation MSE: EWMA / pooled HAR / per-stock HAR': ' / '.join(f"{base_tab.loc[m, 'MSE (log vol)']:.4f}"
                                                                   for m in ['ewma', 'har_pooled', 'har_stock']),
}
display(pd.Series(numbers, name='value').to_frame())

# %% [markdown]
# **§5 summary (for the slides and the revised proposal).**
#
# | Decision | Why |
# |---|---|
# | Target $y_t = \log \text{retvol}$ of month *t*, from row *t+1* only if consecutive; stale `retvol` → missing | log is close to symmetric (§4.1); timing verified in §2.4 |
# | Partial months (run edges, 0.65% of rows) are flagged and kept; `flag_first` may be an input, `flag_tgt_last` may not | GKX has no trading-day count; these rows are harder to forecast, so dropping them would flatter the models |
# | Models use the filled HAR lags `har_3_f`, `har_12_f` plus `har_12_n`; strict lags kept for the textbook HAR | the strict `har_12` would drop 1.84% of rows, which are ~27% more volatile |
# | New inputs: EWMA (λ = 0.5, chosen on validation), `volvol_12`, `rel_vol` / `mkt_lvol` | different memory lengths and a market-relative view of volatility |
# | Target jumps > 5× (0.23% of rows) are flagged and kept | up-jumps are mostly stress months; large down-jumps look like stocks pinned by pending cash acquisitions; real prices, not errors |
# | Every input passes the truncation leakage test | features cannot see the future |
# | Baselines: EWMA, pooled HAR, per-stock HAR (`min_hist = 36`, fall back to pooled); GARCH dropped | pooled HAR is the strongest (validation MSE 0.1256 vs 0.1306 per-stock, 0.1473 EWMA), so ML must beat **pooled** HAR |
# | QLIKE uses the bias-corrected variance $\exp(2\hat y + 2s^2)$; DM test on monthly mean loss differences | fair scoring of log models; cross-sectional correlation handled |
#
# **Hand-off.** Anthony (§6): merge macro and FF5 into `panel_top500_features.parquet`, and keep either `mkt_lvol` or `svar`, not both. Model inputs = `FEATURES` + characteristics + macro; never `LOOK_AHEAD` or `META`.

# %% [markdown]
# ## 6. Market data integration & leakage audit  *(owner: **Anthony** — to do; owns the final notebook)*
#
# **What remains:**
# 1. **Goyal–Welch macro.** Choose the columns that have long, clean histories: `svar, tbl, lty, tms, dfy, dfr, d/p, e/p, b/m, ntis, infl`, plus the short-history `impvar/vrp` as optional. Then apply **publication lags**:
#    * market-price-based series (`svar, tbl, tms, dfy, d/p`) join at month *t−1*;
#    * CPI `infl` joins at *t−2*, because it is released mid-month;
#    * quarterly `cay, i/k` join at the last quarter released by *t−1*.
#    Document each rule in a table.
# 2. **Fama–French 5 factors.** Join month *t−1* returns, plus 3- and 12-month rolling factor volatilities (all ≤ *t−1*).
# 3. **Merge** into one table keyed by (`permno`, `target_month`). Report rows, columns and size in MB, and assert that the row count is unchanged after the merge.
# 4. **Leakage audit of the merged table:**
#    * for every column, state the latest month of information it uses (must be ≤ *t−1*);
#    * a *placebo test*: deliberately leak by shifting each macro variable one month forward (so it uses month *t* data) and check that a simple model's accuracy jumps. This shows the audit can detect a leak, and the correctly aligned version should not show that jump.
# 5. **Scaling and missingness report for macro columns:** ranges, z-scores fitted on the training window only, and which series start late or end before 2021.
# 6. Note that `eqis, cay, i/k, pce, govik, skew, crdstd, accrul, cfacc` are **empty in the monthly sheet**; they live in Goyal–Welch's quarterly and annual sheets. Either drop them or fetch those sheets and forward-fill them with publication lags.
# 7. **Walk-forward split object.** Expanding train from 1980, validation the following year, test target months 2016-01 to 2021-11 (never used for tuning), with a 1-month embargo. Write one generator `walk_forward_splits()` that all modelers import.
#
# **Depends on:** §3 and §5. **Feeds:** every MS3 model.

# %% [markdown]
# ## 7. Sector definitions, data dictionary and revised project framing
#
# This section documents the variables, summarizes the available data and sets out the revised project plan. Industry fields are added to a separate table without changing the data produced by the earlier sections.

# %% [markdown]
# ## 7.1 Input data
#
# This section uses `final_merged` when Section 6 has been completed. Otherwise it uses `feat` from Section 5 and labels the results as before the market-data merge.
#
# New market variables need definitions and timing information in `SECTION6_METADATA`. Variables without this information remain marked as unreviewed. The predictor catalog lists candidates rather than a final model specification.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display

S7_DATA_PATH = None  # Example: Path("Data Clean/panel_top500_features.parquet")
S7_EXPORT_TABLES = False
S7_EXPORT_DIR = (
    Path(OUT) / "Data dictionary"
    if "OUT" in globals()
    else Path("work") / "Data dictionary"
)

s7_input = globals().get("final_merged")
s7_input_stage = "Section 6 merged table"

if not isinstance(s7_input, pd.DataFrame):
    s7_input = globals().get("feat")
    s7_input_stage = "Section 5 feature table, before the Section 6 merge"

s7_input_path = None
if not isinstance(s7_input, pd.DataFrame):
    s7_candidates = (
        [Path(S7_DATA_PATH)]
        if S7_DATA_PATH is not None
        else [
            Path("Data Clean/panel_top500_features.parquet"),
            Path("data_clean/panel_top500_features.parquet"),
        ]
    )
    s7_input_path = next(
        (p for p in s7_candidates if p.is_file()), None
    )

    if s7_input_path is not None:
        s7_input = pd.read_parquet(s7_input_path)
        s7_input_stage = (
            "Explicit local source table; confirm whether Section 6 is included"
        )
    else:
        s7_input = None

if s7_input is None:
    print(
        "No actual feature or merged table is loaded. "
        "Mapping and safeguards are tested below."
    )
    print(
        "Actual-data summaries are not generated. "
        "Set S7_DATA_PATH or run after feat/final_merged."
    )
else:
    print(s7_input_stage)
    print(
        f"Input: {len(s7_input):,} stock-months "
        f"and {len(s7_input.columns):,} columns"
    )
    if s7_input_path is not None:
        print("Source file:", s7_input_path.resolve())

# %% [markdown]
# ## 7.2 Industry classification
#
# The data provide two-digit SIC codes. These are not detailed enough to identify exact Fama-French 12 industries. For example, SIC 2810 belongs to chemicals and SIC 2834 belongs to healthcare, but both appear as `sic2 = 28`.
#
# We therefore use broad SIC divisions. The original `sic2` and `sector` columns stay unchanged. The new `sic_division` column contains the division and `sic_mapping_status` records whether the code was mapped, missing, invalid or nonclassifiable. Missing or unrecognized codes receive `Unknown`. Code 99 is nonclassifiable.
#
# For modeling, treat the industry label as categorical and fit its encoding on training data. Exact FF12 mapping would require more detailed industry information.
#
# Sources: [OSHA SIC manual](https://www.osha.gov/data/sic-manual/) and [Fama-French industry definitions](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_12_ind_port.html).

# %%
S7_DIVISIONS = {
    "Agriculture, forestry and fishing": [1, 2, 7, 8, 9],
    "Mining": [10, 12, 13, 14],
    "Construction": [15, 16, 17],
    "Manufacturing": list(range(20, 40)),
    "Transportation, communications and utilities": list(range(40, 50)),
    "Wholesale trade": [50, 51],
    "Retail trade": list(range(52, 60)),
    "Finance, insurance and real estate": [60, 61, 62, 63, 64, 65, 67],
    "Services": [
        70, 72, 73, 75, 76, 78, 79, 80, 81, 82,
        83, 84, 86, 87, 88, 89
    ],
    "Public administration": list(range(91, 98)),
    "Nonclassifiable": [99],
}

S7_SIC_LOOKUP = {
    code: name
    for name, codes in S7_DIVISIONS.items()
    for code in codes
}
assert len(S7_SIC_LOOKUP) == sum(map(len, S7_DIVISIONS.values()))


def s7_map_sic(values):
    raw = pd.Series(values, copy=True)
    numeric = pd.to_numeric(raw, errors="coerce")
    finite = pd.Series(
        np.isfinite(numeric.astype(float)), index=raw.index
    )
    integer = finite & numeric.mod(1).eq(0)
    boolean_value = raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
    )
    recognized = (
        integer & numeric.isin(S7_SIC_LOOKUP) & ~boolean_value
    )

    division = numeric.map(S7_SIC_LOOKUP).where(
        recognized, "Unknown"
    )
    status = pd.Series(
        "unrecognized_or_invalid", index=raw.index, dtype="object"
    )
    status.loc[raw.isna()] = "missing"
    status.loc[recognized] = "mapped"
    status.loc[recognized & numeric.eq(99)] = "nonclassifiable"

    return pd.DataFrame({
        "sic_division": division,
        "sic_mapping_status": status,
    })


s7_mapping_table = pd.DataFrame([
    {
        "SIC division": name,
        "Valid two-digit codes": ", ".join(
            f"{c:02d}" for c in codes
        ),
    }
    for name, codes in S7_DIVISIONS.items()
])

with pd.option_context("display.max_colwidth", None):
    display(s7_mapping_table)

# %% [markdown]
# ## 7.3 Data dictionary
#
# The dictionary records each variable's meaning, role, timing, type and missingness. The target, identifiers and future-dependent diagnostic flags are not predictor candidates. The older `sector` label is retained for compatibility.
#
# GKX characteristics follow the source's lag convention. The remaining availability assumptions need to be checked for each selected market variable. `mkt_lvol` uses 21 trading days as an approximation. Partial-month flags indicate potential incomplete histories rather than verified IPO or delisting dates.
#
# Added market columns can be documented through `SECTION6_METADATA` with the fields `meaning`, `role`, `latest_information`, `units` and `source`.

# %%
S7_GKX_CHARACTERISTICS = [
    ('absacc', 'Absolute accruals', 'accounting', 'A'),
    ('acc', 'Working capital accruals', 'accounting', 'A'),
    ('aeavol', 'Abnormal earnings announcement volume', 'trading', 'Q'),
    ('age', 'Years since first Compustat coverage', 'firm', 'A'),
    ('agr', 'Asset growth', 'accounting', 'A'),
    ('baspread', 'Bid-ask spread', 'trading', 'M'),
    ('beta', 'Market beta (3y weekly)', 'risk', 'M'),
    ('betasq', 'Beta squared', 'risk', 'M'),
    ('bm', 'Book-to-market', 'valuation', 'A'),
    ('bm_ia', 'Industry-adjusted book-to-market', 'valuation', 'A'),
    ('cash', 'Cash holdings / assets', 'accounting', 'Q'),
    ('cashdebt', 'Cash flow to debt', 'accounting', 'A'),
    ('cashpr', 'Cash productivity', 'accounting', 'A'),
    ('cfp', 'Cash flow to price', 'valuation', 'A'),
    ('cfp_ia', 'Industry-adjusted cash flow to price', 'valuation', 'A'),
    ('chatoia', 'Industry-adj. change in asset turnover', 'accounting', 'A'),
    ('chcsho', 'Change in shares outstanding', 'accounting', 'A'),
    ('chempia', 'Industry-adj. change in employees', 'accounting', 'A'),
    ('chinv', 'Change in inventory', 'accounting', 'A'),
    ('chmom', 'Change in 6-month momentum', 'momentum', 'M'),
    ('chpmia', 'Industry-adj. change in profit margin', 'accounting', 'A'),
    ('chtx', 'Change in tax expense', 'accounting', 'Q'),
    ('cinvest', 'Corporate investment', 'accounting', 'Q'),
    ('convind', 'Convertible debt indicator (0/1)', 'accounting', 'A'),
    ('currat', 'Current ratio', 'accounting', 'A'),
    ('depr', 'Depreciation / PP&E', 'accounting', 'A'),
    ('divi', 'Dividend initiation (0/1)', 'accounting', 'A'),
    ('divo', 'Dividend omission (0/1)', 'accounting', 'A'),
    ('dolvol', 'log dollar trading volume', 'trading', 'M'),
    ('dy', 'Dividend yield', 'valuation', 'A'),
    ('ear', 'Earnings announcement return', 'trading', 'Q'),
    ('egr', 'Growth in common equity', 'accounting', 'A'),
    ('ep', 'Earnings to price', 'valuation', 'A'),
    ('gma', 'Gross profitability', 'accounting', 'A'),
    ('grcapx', 'Growth in capital expenditures', 'accounting', 'A'),
    ('grltnoa', 'Growth in long-term net operating assets', 'accounting', 'A'),
    ('herf', 'Industry sales concentration (Herfindahl)', 'firm', 'A'),
    ('hire', 'Employee growth rate', 'accounting', 'A'),
    ('idiovol', 'Idiosyncratic return volatility (3y weekly)', 'risk', 'M'),
    ('ill', 'Amihud illiquidity', 'trading', 'M'),
    ('indmom', 'Industry momentum', 'momentum', 'M'),
    ('invest', 'Capex and inventory investment', 'accounting', 'A'),
    ('lev', 'Leverage', 'accounting', 'A'),
    ('lgr', 'Growth in long-term debt', 'accounting', 'A'),
    ('maxret', 'Maximum daily return in the month', 'risk', 'M'),
    ('mom12m', '12-month momentum', 'momentum', 'M'),
    ('mom1m', '1-month momentum (last month return)', 'momentum', 'M'),
    ('mom36m', '36-month momentum', 'momentum', 'M'),
    ('mom6m', '6-month momentum', 'momentum', 'M'),
    ('ms', 'Financial statement score (Mohanram, 0-8)', 'accounting', 'Q'),
    ('mvel1', 'Market equity (size, $000), lagged', 'firm', 'M'),
    ('mve_ia', 'Industry-adjusted size', 'firm', 'A'),
    ('nincr', 'Number of consecutive earnings increases (0-8)', 'accounting', 'Q'),
    ('operprof', 'Operating profitability', 'accounting', 'A'),
    ('orgcap', 'Organizational capital', 'accounting', 'A'),
    ('pchcapx_ia', 'Industry-adj. % change in capex', 'accounting', 'A'),
    ('pchcurrat', '% change in current ratio', 'accounting', 'A'),
    ('pchdepr', '% change in depreciation', 'accounting', 'A'),
    ('pchgm_pchsale', '% chg gross margin - % chg sales', 'accounting', 'A'),
    ('pchquick', '% change in quick ratio', 'accounting', 'A'),
    ('pchsale_pchinvt', '% chg sales - % chg inventory', 'accounting', 'A'),
    ('pchsale_pchrect', '% chg sales - % chg receivables', 'accounting', 'A'),
    ('pchsale_pchxsga', '% chg sales - % chg SG&A', 'accounting', 'A'),
    ('pchsaleinv', '% change in sales-to-inventory', 'accounting', 'A'),
    ('pctacc', 'Percent accruals', 'accounting', 'A'),
    ('pricedelay', 'Price delay (speed of reaction to market news)', 'trading', 'M'),
    ('ps', 'Piotroski F-score (0-9)', 'accounting', 'A'),
    ('quick', 'Quick ratio', 'accounting', 'A'),
    ('rd', 'R&D increase (0/1)', 'accounting', 'A'),
    ('rd_mve', 'R&D to market cap', 'valuation', 'A'),
    ('rd_sale', 'R&D to sales', 'accounting', 'A'),
    ('realestate', 'Real estate holdings', 'accounting', 'A'),
    ('retvol', 'Std. dev. of daily returns in the month (realized volatility)', 'risk', 'M'),
    ('roaq', 'Return on assets', 'accounting', 'Q'),
    ('roavol', 'Earnings volatility', 'accounting', 'Q'),
    ('roeq', 'Return on equity', 'accounting', 'Q'),
    ('roic', 'Return on invested capital', 'accounting', 'A'),
    ('rsup', 'Revenue surprise', 'accounting', 'Q'),
    ('salecash', 'Sales to cash', 'accounting', 'A'),
    ('saleinv', 'Sales to inventory', 'accounting', 'A'),
    ('salerec', 'Sales to receivables', 'accounting', 'A'),
    ('secured', 'Secured debt', 'accounting', 'A'),
    ('securedind', 'Secured debt indicator (0/1)', 'accounting', 'A'),
    ('sgr', 'Sales growth', 'accounting', 'A'),
    ('sin', 'Sin stock: alcohol/tobacco/gaming (0/1)', 'firm', 'A'),
    ('sp', 'Sales to price', 'valuation', 'A'),
    ('std_dolvol', 'Volatility of dollar volume', 'trading', 'M'),
    ('std_turn', 'Volatility of share turnover', 'trading', 'M'),
    ('stdacc', 'Accrual volatility', 'accounting', 'Q'),
    ('stdcf', 'Cash flow volatility', 'accounting', 'Q'),
    ('tang', 'Tangibility / debt capacity', 'accounting', 'A'),
    ('tb', 'Tax income to book income', 'accounting', 'A'),
    ('turn', 'Share turnover', 'trading', 'M'),
    ('zerotrade', 'Zero-trading days', 'trading', 'M'),
]


def s7_meta(meaning, role, latest_information, units, source):
    return {
        "meaning": meaning,
        "role": role,
        "latest_information": latest_information,
        "units": units,
        "source": source,
    }


S7_METADATA = {
    name: s7_meta(
        meaning,
        "predictor_candidate",
        "t-1 or earlier under the GKX lag convention",
        "See source definition",
        "GKX dictionary in Section 2",
    )
    for name, meaning, family, frequency in S7_GKX_CHARACTERISTICS
}

S7_METADATA.update({
    "permno": s7_meta(
        "Permanent CRSP security identifier",
        "identifier", "Known before forecast", "ID", "GKX"
    ),
    "date": s7_meta(
        "GKX row date; denotes forecast month t",
        "time_key", "Forecast calendar", "Date", "GKX"
    ),
    "target_month": s7_meta(
        "Month whose realized volatility is forecast",
        "time_key", "Forecast calendar", "Date", "Sections 3 and 5"
    ),
    "sic2": s7_meta(
        "Two-digit SIC industry code; categorical",
        "predictor_candidate",
        "Source row convention, t-1 or earlier",
        "Category", "GKX"
    ),
    "sector": s7_meta(
        "Earlier broad industry label, preserved for compatibility",
        "legacy_metadata", "Source row convention",
        "Category", "Section 3 draft"
    ),
    "sic_division": s7_meta(
        "SIC division from valid two-digit major-group codes",
        "predictor_candidate", "Same availability as sic2",
        "Category", "Section 7 and OSHA SIC manual"
    ),
    "sic_mapping_status": s7_meta(
        "Missing, invalid, nonclassifiable or mapped SIC status",
        "audit_only", "Same availability as sic2",
        "Category", "Section 7"
    ),
    "size_rank": s7_meta(
        "Within-month rank of available lagged market equity used for universe selection",
        "construction_metadata", "t-1", "Rank", "Section 3"
    ),
    "y": s7_meta(
        "Natural log of daily-return standard deviation in forecast month t",
        "target", "t, after forecast",
        "Log decimal daily volatility", "Section 5"
    ),
    "lrv": s7_meta(
        "Natural log of valid same-row retvol; duplicate of har_1",
        "redundant_input", "t-1",
        "Log decimal daily volatility", "Section 5"
    ),
    "har_1": s7_meta(
        "Natural log of previous month's valid retvol",
        "predictor_candidate", "t-1",
        "Log decimal daily volatility", "Section 5"
    ),
    "volvol_12": s7_meta(
        "Sample standard deviation of log retvol in the last 12 rows of a gap-free run; at least 6 valid observations",
        "predictor_candidate", "t-1",
        "Log-volatility standard deviation", "Section 5"
    ),
    "mkt_lvol": s7_meta(
        "0.5 * log(lagged market svar / 21); approximate daily market log volatility",
        "predictor_candidate",
        "t-1 under the market-data availability convention",
        "Log decimal daily volatility", "Section 5 and Goyal-Welch"
    ),
    "rel_vol": s7_meta(
        "har_1 minus mkt_lvol; log stock-to-market volatility ratio",
        "predictor_candidate",
        "t-1 under the market-data availability convention",
        "Log ratio", "Section 5"
    ),
    "flag_first": s7_meta(
        "First observed row of a stock's uninterrupted run; potential partial-history proxy",
        "predictor_candidate", "Known at forecast time",
        "Boolean", "Section 5"
    ),
    "flag_tgt_last": s7_meta(
        "Target read from the last observed row of a run; includes gaps and sample endpoints",
        "evaluation_only", "Future dependent",
        "Boolean", "Section 5"
    ),
    "partial_month": s7_meta(
        "flag_first OR flag_tgt_last; possible partial-month proxy",
        "evaluation_only", "Future dependent",
        "Boolean", "Section 5"
    ),
    "jump5": s7_meta(
        "Absolute change y - har_1 exceeds log(5)",
        "evaluation_only", "Requires target month t",
        "Boolean", "Section 5"
    ),
    "has_next": s7_meta(
        "Next row is a consecutive month for the same stock",
        "construction_metadata", "Future dependent",
        "Boolean", "Section 3"
    ),
    "m": s7_meta(
        "Integer calendar-month key",
        "time_key", "Forecast calendar", "Month index", "Section 3"
    ),
})

for s7_window in (3, 12):
    S7_METADATA[f"har_{s7_window}"] = s7_meta(
        f"0.5 * log(mean of the last {s7_window} monthly retvol squared values)); all observations required within one uninterrupted run",
        "predictor_candidate", "t-1",
        "Log decimal daily volatility", "Section 5"
    )
    S7_METADATA[f"har_{s7_window}_f"] = s7_meta(
        f"0.5 * log(mean of valid retvol squared values in the last {s7_window} rows of a run)); at least one valid observation",
        "predictor_candidate", "t-1",
        "Log decimal daily volatility", "Section 5"
    )
    S7_METADATA[f"har_{s7_window}_n"] = s7_meta(
        f"Number of valid variance observations among the last {s7_window} rows of a run",
        "predictor_candidate", "t-1", "Count", "Section 5"
    )

for s7_lambda in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9):
    S7_METADATA[f"ewma_{round(100*s7_lambda)}"] = s7_meta(
        f"Half log of an EWMA of retvol squared with monthly lambda {s7_lambda}; restarts after gaps and ignores missing observations",
        "predictor_candidate", "t-1",
        "Log decimal daily volatility", "Section 5"
    )

S7_FORBIDDEN_INPUTS = {
    "y", "flag_tgt_last", "partial_month", "jump5", "has_next",
    "permno", "date", "target_month", "m"
}
S7_REQUIRED_METADATA = {
    "meaning", "role", "latest_information", "units", "source"
}


def s7_build_dictionary(frame, extra_metadata=None):
    if not frame.columns.is_unique:
        raise ValueError(
            "Duplicate column names prevent an unambiguous dictionary."
        )

    metadata = {
        name: values.copy()
        for name, values in S7_METADATA.items()
    }

    for name, values in (extra_metadata or {}).items():
        if (
            name in S7_FORBIDDEN_INPUTS
            and values.get("role") == "predictor_candidate"
        ):
            raise ValueError(
                f"{name} is future dependent and cannot be approved as a predictor."
            )

        if S7_REQUIRED_METADATA.issubset(values) and all(
            str(values[k]).strip() for k in S7_REQUIRED_METADATA
        ):
            metadata[name] = dict(values)

    rows = []
    for name in frame.columns:
        entry = metadata.get(
            name,
            s7_meta(
                "Unreviewed added column",
                "unreviewed",
                "Unverified",
                "Unverified",
                "Section 6 metadata required",
            ),
        )
        rows.append({
            "column": name,
            **entry,
            "dtype": str(frame[name].dtype),
            "missing_percent": 100 * frame[name].isna().mean(),
        })

    result = pd.DataFrame(rows).set_index("column")
    approved = set(
        result.index[result.role.eq("predictor_candidate")]
    )

    if approved & S7_FORBIDDEN_INPUTS:
        raise ValueError(
            "The candidate catalog contains a future-dependent field."
        )

    return result


# %% [markdown]
# ## 7.4 Data summary
#
# The checks below preserve the input rows and verify the stock-month keys and target dates. They add the two industry fields to a separate table.
#
# Numeric summary statistics use development data from 1980 to 2015. Population counts cover each time window. Some earlier descriptive analyses include 2016 to 2021, so that period is not described as completely unseen. Model selection must still use development data only.

# %%
def s7_prepare_table(frame):
    required = {"permno", "date", "target_month", "sic2", "y"}

    if not required.issubset(frame.columns):
        raise ValueError(
            f"Missing required columns: {sorted(required - set(frame.columns))}"
        )
    if not frame.columns.is_unique:
        raise ValueError("Duplicate column names in source table.")

    result = frame.copy()
    date = pd.to_datetime(result["date"], errors="raise")
    target = pd.to_datetime(result["target_month"], errors="raise")

    if (
        result["permno"].isna().any()
        or date.isna().any()
        or target.isna().any()
    ):
        raise ValueError("Stock and time keys must not be missing.")

    if not date.dt.to_period("M").equals(target.dt.to_period("M")):
        raise ValueError(
            "Forecast target_month must equal the GKX row's calendar month."
        )

    keys = pd.DataFrame({
        "permno": result["permno"],
        "month": target.dt.to_period("M"),
    })
    if keys.duplicated().any():
        raise ValueError(
            "Duplicate stock-month rows; inspect the merge before summarizing."
        )

    target_values = pd.to_numeric(result["y"], errors="raise")
    if np.isinf(target_values.astype(float)).any():
        raise ValueError(
            "Infinite target values are not valid log volatility."
        )

    mapped = s7_map_sic(result["sic2"])
    result["sic_division"] = mapped["sic_division"].to_numpy()
    result["sic_mapping_status"] = mapped["sic_mapping_status"].to_numpy()

    assert len(result) == len(frame)
    assert result.index.equals(frame.index)
    return result


def s7_population_summary(frame):
    target = pd.to_datetime(frame["target_month"])
    windows = {
        "All supplied rows": pd.Series(True, index=frame.index),
        "Modeling window 1980-01 to 2021-11": target.between(
            "1980-01-01", "2021-11-30"
        ),
        "Training 1980-2010": target.between(
            "1980-01-01", "2010-12-31"
        ),
        "Validation 2011-2015": target.between(
            "2011-01-01", "2015-12-31"
        ),
        "Reserved evaluation 2016-01 to 2021-11": target.between(
            "2016-01-01", "2021-11-30"
        ),
    }

    rows = []
    for name, selected in windows.items():
        part = frame.loc[selected]
        dates = target.loc[selected]
        rows.append({
            "population": name,
            "stock_months": len(part),
            "stocks": part["permno"].nunique(),
            "observed_targets": part["y"].notna().sum(),
            "first_month": (
                str(dates.min().to_period("M")) if len(part) else ""
            ),
            "last_month": (
                str(dates.max().to_period("M")) if len(part) else ""
            ),
        })

    return pd.DataFrame(rows).set_index("population")


def s7_numeric_summary(frame, dictionary):
    numeric = frame.select_dtypes(
        include=["number", "bool"]
    ).astype(float)

    if numeric.empty:
        return pd.DataFrame()

    finite = numeric.replace([np.inf, -np.inf], np.nan)
    summary = finite.describe(percentiles=[0.25, 0.5, 0.75]).T

    summary.insert(
        0, "role", dictionary["role"].reindex(summary.index)
    )
    summary.insert(
        1, "missing_percent", 100 * numeric.isna().mean()
    )
    summary.insert(
        2, "nonfinite_count", np.isinf(numeric).sum()
    )
    return summary


s7_table = None
s7_dictionary = None

if s7_input is not None:
    s7_table = s7_prepare_table(s7_input)
    s7_dictionary = s7_build_dictionary(
        s7_table, globals().get("SECTION6_METADATA", {})
    )
    s7_population = s7_population_summary(s7_table)

    s7_dates = pd.to_datetime(s7_table["target_month"])
    s7_development = s7_table.loc[
        s7_dates.between("1980-01-01", "2015-12-31")
    ]
    s7_summary = s7_numeric_summary(
        s7_development, s7_dictionary
    )

    s7_sector_counts = (
        s7_development["sic_division"]
        .value_counts(dropna=False)
        .rename("stock_months")
        .to_frame()
    )
    s7_sector_counts["share_percent"] = (
        100 * s7_sector_counts["stock_months"]
        / len(s7_development)
    )

    s7_unreviewed = s7_dictionary.loc[
        s7_dictionary.role.eq("unreviewed")
    ]
    s7_candidate_columns = s7_dictionary.index[
        s7_dictionary.role.eq("predictor_candidate")
    ].tolist()

    print(s7_input_stage)
    print(
        f"Input columns: {s7_input.shape[1]}; "
        f"Section 7 view columns: {s7_table.shape[1]} "
        "(division and mapping-status columns)"
    )
    print(
        f"Input memory: "
        f"{s7_input.memory_usage(index=True, deep=True).sum() / 2**20:.2f} MiB; "
        "not file size"
    )

    display(s7_population)
    display(s7_sector_counts.round(2))

    print(
        "Industry mapping audit:",
        s7_table.sic_mapping_status.value_counts().to_dict(),
    )
    print(
        f"Documented candidate columns: {len(s7_candidate_columns)}; "
        f"unreviewed columns: {len(s7_unreviewed)}"
    )

    if len(s7_unreviewed):
        display(s7_unreviewed)

    summary_columns = [
        c for c in [
            "y", "har_1", "har_3_f", "har_12_f",
            "har_12_n", "ewma_50", "volvol_12", "rel_vol"
        ]
        if c in s7_summary.index
    ]
    display(s7_summary.loc[summary_columns].round(4))

else:
    print(
        "Actual-data audit and summary not run "
        "because the input table is not available."
    )

# %% [markdown]
# ### Full dictionary and summary statistics
#
# The dictionary describes all supplied columns. Its missingness percentages use the full input. Numeric summary statistics use development data only. Identifiers are included for documentation and are labeled separately from predictors.

# %%
if s7_table is not None:
    with pd.option_context(
        "display.max_rows", None,
        "display.max_colwidth", None,
    ):
        display(s7_dictionary)
        display(s7_summary.round(6))

else:
    s7_known_dictionary = (
        pd.DataFrame.from_dict(S7_METADATA, orient="index")
        .rename_axis("column")
    )
    print(
        "Definitions for the known Section 3-5 columns are available below; "
        "observed dtypes and statistics require data."
    )

    with pd.option_context(
        "display.max_rows", 30,
        "display.max_colwidth", None,
    ):
        display(s7_known_dictionary.loc[
            [
                "y", "sic_division", "har_1", "har_3_f", "har_12_f",
                "har_12_n", "ewma_50", "volvol_12", "mkt_lvol",
                "rel_vol", "flag_first", "flag_tgt_last",
                "partial_month", "jump5"
            ]
        ])

if S7_EXPORT_TABLES and s7_table is not None:
    S7_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    for filename, table in {
        "section7_column_dictionary.csv": s7_dictionary,
        "section7_development_numeric_summary.csv": s7_summary,
        "section7_population_counts.csv": s7_population,
        "section7_development_sector_summary.csv": s7_sector_counts,
    }.items():
        destination = S7_EXPORT_DIR / filename

        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing table: {destination}"
            )

        table.to_csv(destination, encoding="utf-8-sig")

    print("Saved local tables to", S7_EXPORT_DIR.resolve())

# %% [markdown]
# ## 7.5 Baseline results
#
# The table below records the results reported in Section 5. The feature-table dimensions are from before the market-data merge. These figures need to be refreshed if the underlying data or inclusion rules change.

# %%
s7_recorded_results = pd.DataFrame([
    (
        "Feature export before Section 6",
        "388,502 rows × 121 columns",
        "Section 5.8 saved output",
    ),
    (
        "Target-valid modeling stock-months from 1980",
        "250,719",
        "Section 5.8 saved output",
    ),
    (
        "Strict 12-month HAR warm-up exclusions",
        "4,601 rows (1.84%)",
        "Section 5.4 saved output",
    ),
    (
        "Candidate monthly EWMA decay selected on validation",
        "0.5",
        "Section 5.5 saved output",
    ),
    (
        "Validation MSE, raw EWMA",
        "0.1473",
        "Section 5.7; 2011-2015, common rows",
    ),
    (
        "Validation MSE, pooled HAR",
        "0.1256",
        "Section 5.7; 2011-2015, common rows",
    ),
    (
        "Validation MSE, stock HAR with pooled fallback",
        "0.1306",
        "Section 5.7; 2011-2015, common rows",
    ),
], columns=["Quantity", "Recorded value", "Provenance"])

with pd.option_context("display.max_colwidth", None):
    display(s7_recorded_results)


# %% [markdown]
# ## 7.6 Revised question and plan
#
# Can pooled regularized regression and gradient-boosted trees improve forecasts of next-month log realized volatility beyond a pooled HAR benchmark for the 500 largest eligible U.S. stocks selected each month using lagged market equity?
#
# The available data are monthly. We therefore replace the original weekly Garman-Klass target with log volatility based on GKX `retvol`. The data do not contain the daily high and low prices needed for the original target. The selected stocks form a large-cap proxy rather than historical S&P 500 membership.
#
# We plan to compare ridge, lasso, elastic net and gradient-boosted trees with pooled HAR. Stock-specific HAR with a pooled fallback and EWMA provide additional benchmarks. Comparisons will use the same eligible observations and chronological validation. Preprocessing will be fitted within the training data.
#
# We will evaluate log-volatility MSE and variance QLIKE. The conversion from predicted log volatility to variance will state its assumptions. Forecast-loss differences will be aggregated by month before calculating uncertainty that accounts for dependence. The planned validation procedure includes a one-month gap. The Section 5 scores use the earlier boundary and must be recomputed before comparison under that procedure.
#
# The project focuses on forecast accuracy. Portfolio backtests and Sharpe ratios are outside the revised scope. Regularized regression and gradient boosting are the priorities. Clustering and neural networks are optional. GKX includes historical securities, so a new yfinance download is unnecessary. This reduces dependence on a current surviving-stock list but does not remove every possible selection bias.
#
# For the AC209A extension, we propose the adapted HAR framework and Diebold-Mariano forecast comparisons with uncertainty estimates that account for heteroskedasticity and autocorrelation. The supporting readings are Corsi (2009) and Diebold and Mariano (1995).
#
# Once Section 6 is complete, the data summary and presentation figures should be checked against its selected variables and merged table.

# %% [markdown]
# ## 7.7 Checks
#
# The small synthetic example below checks industry mapping, row preservation, duplicate keys, target dates, variable roles and summary calculations. It is only a test and does not supply any project results.

# %%
def s7_expect_error(function, message):
    try:
        function()
    except ValueError:
        return
    raise AssertionError(message)


s7_fixture = pd.DataFrame({
    "permno": [101, 101, 202, 202],
    "date": pd.to_datetime([
        "2010-11-30", "2010-12-31", "2011-01-31", "2011-02-28"
    ]),
    "target_month": pd.to_datetime([
        "2010-11-30", "2010-12-31", "2011-01-31", "2011-02-28"
    ]),
    "sic2": [28.0, np.nan, 99.0, 28.5],
    "sector": ["old"] * 4,
    "y": [-4.1, -3.8, -4.0, -3.9],
    "har_1": [-4.0, -4.1, -3.9, -4.0],
    "partial_month": [False, True, False, False],
    "jump5": [False] * 4,
    "new_unreviewed_macro": [1.0, 2.0, 3.0, 4.0],
}, index=[8, 2, 5, 1])

s7_original_fixture = s7_fixture.copy(deep=True)
s7_checked = s7_prepare_table(s7_fixture)

pd.testing.assert_frame_equal(
    s7_fixture, s7_original_fixture
)
assert s7_checked.index.tolist() == [8, 2, 5, 1]
assert s7_checked.sic_division.tolist() == [
    "Manufacturing", "Unknown", "Nonclassifiable", "Unknown"
]
assert s7_checked.sic_mapping_status.tolist() == [
    "mapped", "missing", "nonclassifiable", "unrecognized_or_invalid"
]

s7_bad_codes = s7_map_sic(pd.Series([
    0, 3, 11, 18, 66, 68, 71, 85, 90, 98,
    100, np.inf, "bad", True
]))
assert s7_bad_codes.sic_division.eq("Unknown").all()

s7_dict_test = s7_build_dictionary(s7_checked)
assert (
    s7_dict_test.loc["new_unreviewed_macro", "role"]
    == "unreviewed"
)
assert (
    s7_dict_test.loc["partial_month", "role"]
    == "evaluation_only"
)

s7_expect_error(
    lambda: s7_build_dictionary(
        s7_checked,
        {"y": {"role": "predictor_candidate"}},
    ),
    "Target override was not rejected",
)

s7_expect_error(
    lambda: s7_prepare_table(
        pd.concat([s7_fixture, s7_fixture.iloc[[0]]])
    ),
    "Duplicate key was not rejected",
)

s7_wrong_month = s7_fixture.copy()
s7_wrong_month.loc[8, "target_month"] = pd.Timestamp(
    "2010-12-31"
)

s7_expect_error(
    lambda: s7_prepare_table(s7_wrong_month),
    "Wrong target month was not rejected",
)

s7_test_population = s7_population_summary(s7_checked)
assert (
    s7_test_population.loc["Training 1980-2010", "stock_months"]
    == 2
)
assert (
    s7_test_population.loc["Validation 2011-2015", "stock_months"]
    == 2
)

s7_test_summary = s7_numeric_summary(
    s7_checked, s7_dict_test
)
assert np.isclose(
    s7_test_summary.loc["y", "mean"], -3.95
)
assert np.isclose(
    s7_test_summary.loc["sic2", "missing_percent"], 25.0
)

print(
    "Synthetic checks passed: mapping, row preservation, "
    "keys, timing, roles and summary arithmetic."
)

# %% [markdown]
# ## 8. Revised project question (draft for the group)
#
# **Main question (forecasting, regression).**
# * **Setting:** the 500 largest U.S. stocks each month, 1980–2021, with information available at the end of month *t−1*.
# * **Question:** does a **pooled** model that adds cross-sectional and market-state features predict the stock's **log realized volatility in month *t*** better than a **per-stock HAR(1, 3, 12-month)** model?
# * **Candidate pooled models:** ridge, lasso and elastic net on HAR + characteristics + macro; gradient-boosted trees.
# * **Test:** out of sample on target months 2016-01 to 2021-11, measured by MSE on log volatility and QLIKE on variance. Significance is assessed at the 5% level with a Diebold–Mariano test on the monthly cross-sectional average loss differential.
#
# **Secondary questions.**
# 1. Which features account for any improvement (permutation importance, lasso selection stability across bootstraps)?
# 2. Does the improvement hold separately in calm months and in stress months, where stress means `svar` is above its training-period 90th percentile? We report block-bootstrap confidence intervals, plus a secondary 2008–09 out-of-sample check.
#
# **How we will know it was answered:** the DM-test p-value and the loss ratio (model ÷ HAR) on the fixed test set, reported for every model we try.
#
# *Changed from the original proposal:*
# * the horizon moves from week to month and the data from daily yfinance to monthly GKX, because no daily OHLC data are available;
# * the test window moves from 2019–2026 to 2016–2021, because the data end in 2021;
# * the economic-value backtest is dropped, as the staff required.
