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
#     display_name: cs209a-vol-forecast (3.13.5)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # CS1090A / AC209A — Milestone 2: Data Wrangling and Project Redefinition
#
# **Canvas project group #: 7** · **Project #12 — Forecasting Stock Volatility on the S&P 500: Can Machine Learning Beat HAR?**
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
# | 6 | Market data integration & leakage audit (Goyal–Welch macro, FF5, quarterly/annual GW, merged table, walk-forward splits) | **Anthony** | code ✔ |
# | 7 | Sector labels, data dictionary, staff caveats, revised proposal & slides | **Nic** | text ✔ (proposal and slides are separate PDFs) |
# | 8 | Revised project question | all | text ✔ |
#
# > **One-line summary.** One row of our final table is *one large U.S. stock in one month*. From everything known at the end of month *t−1*, we predict the log of that stock's realized volatility in month *t*. The questions are whether pooled ML models (ridge/lasso/elastic net, gradient-boosted trees) beat the **pooled HAR** benchmark (§5.7, §8), which features drive any gain, and whether the gain survives in stress periods.
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
    DATA, OUT = os.path.join('.', 'Data'), 'data_clean'
assert os.path.isdir(DATA), f'{DATA} not found (Colab: add final_project as a shortcut to My Drive; locally: put the data in ./Data)'
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
# | `Data2025.xlsx - Quarterly.csv` | Goyal & Welch predictor data (2025 update, "Quarterly" sheet), https://sites.google.com/view/agoyal145 | quarterly, 1871-Q1 → 2025-Q4 | optional slow block (§6.2b): `i/k`, `govik`, `crdstd` |
# | `Data2025.xlsx - Annual.csv` | Goyal & Welch predictor data (2025 update, "Annual" sheet), https://sites.google.com/view/agoyal145 | annual, 1871 → 2025 | optional slow block (§6.2b): `eqis`, `accrul`, `cfacc` |
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
# the 3.6 GB raw csv is unzipped locally (datashare.csv) but stored only zipped on Drive (datashare.zip)
raw = {f: os.path.join(dp, f) for dp, _, fs in os.walk(DATA) for f in fs if f in ('datashare.csv', 'datashare.zip')}
if 'datashare.csv' in raw:
    GKX_ZIP, GKX_CSV = None, raw['datashare.csv']
    print(f'raw GKX csv: {os.path.getsize(GKX_CSV)/2**30:.2f} GB uncompressed')
else:                                                   # read the uncompressed size from the zip header (no extraction)
    GKX_ZIP = raw['datashare.zip']
    with zipfile.ZipFile(GKX_ZIP) as z:
        GKX_CSV = [n for n in z.namelist() if n.endswith('.csv')][0]
        print(f'raw GKX csv inside datashare.zip: {z.getinfo(GKX_CSV).file_size/2**30:.2f} GB uncompressed '
              f'({z.getinfo(GKX_CSV).compress_size/2**30:.2f} GB zipped)')

# %%
t0 = time.time()
gkx = pd.read_parquet(os.path.join(DATA, 'datashare_small.parquet'))
gkx['date'] = pd.to_datetime(gkx['date'])
# the parquet must contain every row of the raw csv (count newlines without parsing 3.8 GB)
CHECK_ROWS = True                     # streams 3.6 GB (a few minutes on Drive, from the zip); False to skip
if CHECK_ROWS:
    with (open(GKX_CSV, 'rb') if GKX_ZIP is None else zipfile.ZipFile(GKX_ZIP).open(GKX_CSV)) as fh:
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
# ## 6. Market data integration & leakage audit  *(owner: **Anthony**; owns the final notebook)*
#
# **What remains:**
# 1. **Goyal–Welch macro.** Choose the columns that have long, clean histories: `svar, tbl, lty, tms, dfy, dfr, d/p, e/p, b/m, ntis, infl`, plus the short-history `impvar/vrp` as optional. Then apply **publication lags**:
#    * market-price-based series (`svar, tbl, tms, dfy, d/p`) join at month *t−1*;
#    * CPI `infl` joins at *t−2*, because it is released mid-month;
#    * quarterly `cay, i/k` join at the last quarter released by *t−1*.
#    Document each rule in a table.
# 2. **Fama–French 5 factors.** Join month *t−1* returns, plus 3- and 12-month rolling factor volatilities (all ≤ *t−1*).
# 3. **Merge** into one table keyed by (`permno`, `target_month`). Report rows, columns and size in MB, and assert that the row count is unchanged after the merge. *(Done in §6.3, on Cathy's `panel_top500_features.parquet`.)*
# 4. **Leakage audit of the merged table:**
#    * for every column, state the latest month of information it uses (must be ≤ *t−1*);
#    * a *placebo test*: deliberately leak by shifting each macro variable one month forward (so it uses month *t* data) and check that a simple model's accuracy jumps. This shows the audit can detect a leak, and the correctly aligned version should not show that jump. *(Done in §6.4: `MODEL_INPUTS`, audit table and placebo test.)*
# 5. **Scaling and missingness report for macro columns:** ranges, z-scores fitted on the training window only, and which series start late or end before 2021. *(Done in §6.5: report, `fold_zscore()` (§6.0), `lty` dropped.)*
# 6. Note that `eqis, cay, i/k, pce, govik, skew, crdstd, accrul, cfacc` are **empty in the monthly sheet**; they live in Goyal–Welch's quarterly and annual sheets. Either drop them or fetch those sheets and forward-fill them with publication lags. *(Done in §6.2b from Data2025.xlsx: `i/k, govik, crdstd, eqis, accrul, cfacc` kept as an optional block; `cay, pce, skew` excluded.)*
# 7. **Walk-forward split object.** Expanding train from 1980, validation the following year, test target months 2016-01 to 2021-11 (never used for tuning), with a 1-month embargo. Write one generator `walk_forward_splits()` that all modelers import. *(Done in §6.7 / the split functions in §6.0: expanding window, annual refit in the test period, `mode='tune'|'test'`.)*
#
# **Depends on:** §3 and §5. **Feeds:** every MS3 model.

# %% [markdown]
# ### 6.0 Walk-forward split functions *(shared by every model)*
#
# These functions were kept in a separate `splits.py`. They are defined here so that the notebook runs on its own.
#
# * `walk_forward_splits(df, mode='tune' | 'test')` yields one fold per year. Training always starts in 1980-01 and expands by one year per fold, and a **1-month embargo** separates the last training target month from the first evaluation month. `mode='tune'` validates on 2011, …, 2015; `mode='test'` refits before each test year 2016, …, 2021 (test ends 2021-11).
# * `describe_splits(df)` lists every fold's date ranges and row counts (shown in §6.7).
# * `fold_zscore(df, cols, train_idx)` z-scores columns with **one fold's training rows only** and adds `<col>_missing` indicators (used in §6.5).
#
# Each fold returns positional row indices (`train_idx`, `eval_idx`), and built-in assertions check that training and evaluation rows never overlap, that the embargo holds, and that tuning never touches the test window.

# %%
# Walk-forward splits shared by every model (formerly splits.py). Defined once here and used in
# §6.4b (placebo test), §6.5 (per-fold z-scores) and §6.7 (fold table). MS3 models reuse these functions.
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



# %%
gw = pd.read_csv(os.path.join(DATA, 'Data2024.xlsx - Monthly.csv'))
gw['ym'] = pd.PeriodIndex(gw['yyyymm'].astype(str), freq='M')
gw = gw.set_index('ym').drop(columns='yyyymm').apply(pd.to_numeric, errors='coerce')
print(f'Goyal-Welch: {gw.shape[0]:,} months x {gw.shape[1]} cols, {gw.index.min()} -> {gw.index.max()}')

display(gw)

# %% [markdown]
# ### 6.1 Goyal–Welch macro: publication lags
#
# A Goyal–Welch row for month *m* is used to forecast target month *m + lag*. A panel row dated *t* holds information up to the end of month *t−1*, so every lag is **at least 1**.
#
# * **Market-price series** (`svar, tbl, lty, tms, dfy, dfr, d/p, b/m, ntis`, and optional `impvar, vrp`): lag 1. They are known at the end of the month they describe. `b/m` is safe at lag 1 because Goyal–Welch already divide by the *previous* year's book value (two years back in January–February).
# * **`infl`**: lag 2. CPI for month *m* is released in the middle of month *m+1*.
# * **`e/p`**: rebuilt as `e12` at lag 5 divided by `price` at lag 1. `e12` changes every month even though S&P earnings are quarterly: the monthly values are **interpolated between quarter-ends** (the check below shows equal monthly steps), so a month's value already contains the coming quarter-end's earnings, which are published about two months after the quarter ends. Lag 5 covers both delays. The GW-provided `e/p` column at lag 1 would leak.
# * `cay, i/k` (quarterly) are empty in the monthly sheet and are dropped (see item 6).

# %%
# Publication lags: GW value for month m is used for target month m + lag.
# Panel row t = info at end of t-1, so lag >= 1 for everything.
GW_RULES = {
    # col      lag  reason
    'svar':   (1, 'sum of squared daily S&P returns in month m; known at month end'),
    'tbl':    (1, '3-month T-bill yield, market price'),
    'lty':    (1, 'long-term govt yield, market price'),
    'tms':    (1, 'lty - tbl'),
    'dfy':    (1, 'BAA - AAA yields'),
    'dfr':    (1, 'corporate minus govt bond return over month m'),
    'd/p':    (1, 'trailing 12m dividends / price'),
    'b/m':    (1, 'GW already use prior-year book value (2 yrs back in Jan-Feb)'),
    'ntis':   (1, 'NYSE net issuance, 12m moving sum'),
    'infl':   (2, 'CPI for month m is released mid-month m+1'),
    'impvar': (1, 'VIX-based implied variance (1996+, optional)'),
    'vrp':    (1, 'variance risk premium (1990+, optional)'),
}
EP_LAG = 5   # e12 is interpolated within quarters + ~2-month reporting delay

# Evidence that e12 is interpolated: equal monthly steps inside a quarter
print('e12 monthly changes, 2021-10..2022-03:', gw.loc['2021-10':'2022-03', 'e12'].diff().round(4).tolist())

def lag_series(s, k):
    s = s.copy(); s.index = s.index + k          # PeriodIndex + k = k months later
    return s

def build_macro(gw):
    mac = pd.DataFrame({c: lag_series(gw[c], k) for c, (k, _) in GW_RULES.items()})
    mac['e/p'] = lag_series(gw['e12'], EP_LAG) / lag_series(gw['price'], 1)   # stale earnings, fresh price
    mac['log_svar'] = np.log(mac['svar'].where(mac['svar'] > 0))   # svar = 0 only Aug-Nov 1914 (NYSE closed)
    mac.index.name = 'target_ym'
    return mac

macro = build_macro(gw)

# Documentation table for the notebook
lag_table = pd.DataFrame([(c, f't-{k}', why) for c, (k, why) in GW_RULES.items()]
                         + [('e/p', f't-{EP_LAG} (e12), t-1 (price)', 'earnings interpolated within quarter + reporting delay')],
                         columns=['column', 'latest month used', 'reason'])
lag_table.to_csv(os.path.join(DOCS, 'goyal_welch_publication_lags.csv'), index=False, encoding='utf-8-sig')
display(lag_table)

# Spot checks: the value for target month 2020-03 must come from the right GW month
assert macro.loc['2020-03', 'svar'] == gw.loc['2020-02', 'svar']
assert macro.loc['2020-03', 'infl'] == gw.loc['2020-01', 'infl']
assert np.isclose(macro.loc['2020-03', 'e/p'], gw.loc['2019-10', 'e12'] / gw.loc['2020-02', 'price'])

win = macro.loc['1980-01':'2021-11']
print(f'macro table: {macro.shape[1]} columns; missing target months in 1980-01..2021-11:')
display(win.isna().sum().to_frame('missing months').T)

# %% [markdown]
# ### 6.2 Fama–French 5 factors
#
# Factor returns for month *m* are computed from month-*m* prices, so they are known at the end of month *m* and join at **lag 1**, like the market-price series in §6.1. `RF` is dropped because it duplicates the Goyal–Welch T-bill rate.
#
# For each factor we keep:
# * last month's return (`ff_mkt`, `ff_smb`, …);
# * its rolling volatility over the last 3 and 12 months (`ff_mkt_vol3`, `ff_mkt_vol12`, …), all ending at *t−1*.
#
# The rolling volatility is the **root mean square** of monthly returns, not the sample standard deviation. With only 3 observations, subtracting the sample mean wastes a degree of freedom, and the mean monthly factor return is close to 0 anyway. This is the same convention as `svar` (sum of squared daily returns, no demeaning).

# %%
FF_MAP = {'Mkt-RF': 'mkt', 'SMB': 'smb', 'HML': 'hml', 'RMW': 'rmw', 'CMA': 'cma'}   # RF dropped (= GW tbl)
FF_LAG = 1
assert len(ff) == (ff.index[-1] - ff.index[0]).n + 1, 'FF months must be consecutive for positional rolling windows'

def build_ff_features(ff):
    feat = {}
    for src, name in FF_MAP.items():
        r = ff[src]
        feat[f'ff_{name}'] = r
        for w in (3, 12):
            feat[f'ff_{name}_vol{w}'] = np.sqrt((r ** 2).rolling(w, min_periods=w).mean())   # RMS, ends at month m
    out = lag_series(pd.DataFrame(feat), FF_LAG)
    out.index.name = 'target_ym'
    return out

ff_feat = build_ff_features(ff)

# Spot checks for target month 2020-03: last month = 2020-02, 3-month window = 2019-12..2020-02
assert ff_feat.loc['2020-03', 'ff_mkt'] == ff.loc['2020-02', 'Mkt-RF']
assert np.isclose(ff_feat.loc['2020-03', 'ff_mkt_vol3'], np.sqrt((ff.loc['2019-12':'2020-02', 'Mkt-RF'] ** 2).mean()))

ff_rules = pd.DataFrame([(f'ff_{n}', 't-1', f'{s} return over month t-1') for s, n in FF_MAP.items()]
                        + [(f'ff_{n}_vol{w}', 't-1', f'RMS of {s} returns, months t-{w}..t-1') for s, n in FF_MAP.items() for w in (3, 12)],
                        columns=['column', 'latest month used', 'reason'])
lag_table = pd.concat([lag_table[~lag_table.column.isin(ff_rules.column)], ff_rules], ignore_index=True)   # rerun-safe
lag_table.to_csv(os.path.join(DOCS, 'market_data_publication_lags.csv'), index=False, encoding='utf-8-sig')

# How much do the market-factor vols overlap with GW's daily-based svar? (target months 1980..VAL_END only)
chk = pd.DataFrame({'log sqrt(svar)': 0.5 * macro['log_svar'],
                    'log ff_mkt_vol3': np.log(ff_feat['ff_mkt_vol3']),
                    'log ff_mkt_vol12': np.log(ff_feat['ff_mkt_vol12']),
                    '|ff_mkt|': ff_feat['ff_mkt'].abs()}).loc['1980-01':VAL_END.strftime('%Y-%m')]
print('Correlation of lagged market-vol measures, target months 1980 to end of validation:')
display(chk.corr().round(2))

market = macro.join(ff_feat, how='left')                     # one row per target month: GW + FF5
print(f'market table: {market.shape[1]} columns; missing target months in 1980-01..2021-11:')
display(market.loc['1980-01':'2021-11'].isna().sum().loc[lambda s: s > 0].to_frame('missing months').T)

# %% [markdown]
# ### 6.2b Quarterly and annual Goyal–Welch series *(plan item 6; optional feature block)*
#
# These columns are empty in the monthly sheet. They come from the Quarterly and Annual sheets of Goyal's newer **Data2025.xlsx** (saved as `Data2025.xlsx - Quarterly.csv` / `- Annual.csv`). Data2025's ReadMe sheet decides which are safe:
#
# | Kept | Frequency | Rule |
# |---|---|---|
# | `i/k`, `govik`, `crdstd` | quarterly | a quarter's value is used from **3 months after the quarter ends** (known by the end of the 2nd month after) |
# | `eqis`, `accrul`, `cfacc` | annual | a year's value is used from **7 months after the year ends**, i.e. from July of the next year (same 6-month delay GKX use for annual data) |
#
# Each value is then held until the next one arrives, and never longer than one period.
#
# **Excluded:**
# * `cay`, `pce`: the ReadMe says they are full-sample estimates ("needs recomputation every period"), so every value uses data up to 2025 and would leak.
# * `skew`: semiannual and ends in 2019, so it is missing for most of the test window.
# * Monthly `ogap, sntm, fbm, tchi, shtint`: the ReadMe flags them as full-sample too. They are not in §6.1, and are listed here so nobody adds them later.
#
# **Caveats.**
# * `i/k` and `govik` are latest-vintage national-accounts data, so later revisions leak in slightly. This data can't avoid that.
# * `crdstd` starts in 1990, so it is missing for 1980–1990 of the training window.
# * Annual series change once a year, so tree models can use them as era labels. Treat `SLOW_COLS` as an **optional block**: keep it only if validation loss improves.

# %%
Q_PATH, A_PATH = (os.path.join(DATA, f'Data2025.xlsx - {k}.csv') for k in ('Quarterly', 'Annual'))
HAVE_SLOW = os.path.exists(Q_PATH) and os.path.exists(A_PATH)
if not HAVE_SLOW:   # optional block: skip cleanly if the Data2025 sheets are not in DATA
    print('Data2025 Quarterly/Annual csv not found in DATA -> optional slow block skipped (SLOW_COLS = [])')
    SLOW_COLS, Q_COLS, A_COLS = [], [], []
else:
    gwq, gwa = pd.read_csv(Q_PATH), pd.read_csv(A_PATH)
    Q_COLS, A_COLS = ['i/k', 'govik', 'crdstd'], ['eqis', 'accrul', 'cfacc']
    SLOW_COLS = Q_COLS + A_COLS
    Q_AVAIL, A_AVAIL = 3, 7          # first target month allowed = period-end month + this many months

    def slow_series(df, key, cols, freq, avail, hold):
        """Put period values at the first target month allowed to use them, then hold them for at most `hold` more months."""
        k = df[key].astype(int)
        if freq == 'Q':                                                  # yyyyq, e.g. 19801 = 1980 Q1
            end = pd.PeriodIndex(((k // 10).astype(str) + 'Q' + (k % 10).astype(str)).values, freq='Q').asfreq('M', 'end')
        else:                                                            # yyyy
            end = pd.PeriodIndex(k.astype(str).values, freq='Y').asfreq('M', 'end')
        vals = df[cols].apply(pd.to_numeric, errors='coerce').set_axis(end + avail)
        months = pd.period_range(vals.index.min(), market.index.max(), freq='M')
        return vals.reindex(months).ffill(limit=hold)

    slow = pd.concat([slow_series(gwq, 'yyyyq', Q_COLS, 'Q', Q_AVAIL, hold=2),
                      slow_series(gwa, 'yyyy', A_COLS, 'A', A_AVAIL, hold=11)], axis=1)
    slow.index.name = 'target_ym'

    # Spot checks: target 1980-06 may use 1980 Q1 (ends Mar), 1980-05 only 1979 Q4; 1980-07 may use year 1979, 1980-06 only 1978
    qv = gwq.set_index('yyyyq'); av = gwa.set_index('yyyy')
    assert slow.loc['1980-06', 'i/k'] == qv.loc[19801, 'i/k'] and slow.loc['1980-05', 'i/k'] == qv.loc[19794, 'i/k']
    assert slow.loc['1980-07', 'eqis'] == av.loc[1979, 'eqis'] and slow.loc['1980-06', 'eqis'] == av.loc[1978, 'eqis']

    market = market.drop(columns=SLOW_COLS, errors='ignore').join(slow, how='left')   # rerun-safe

    slow_rules = pd.DataFrame([(c, f'quarter ending <= t-{Q_AVAIL}', 'quarterly national accounts / loan-officer survey, release delay') for c in Q_COLS]
                              + [(c, f'year ending <= t-{A_AVAIL}', 'annual data, 6-month delay as in GKX') for c in A_COLS]
                              + [(c, 'excluded', 'full-sample estimate in Data2025 ReadMe (look-ahead)') for c in ['cay', 'pce', 'ogap', 'sntm', 'fbm', 'tchi', 'shtint']]
                              + [('skew', 'excluded', 'ends 2019, missing in test window')],
                              columns=['column', 'latest month used', 'reason'])
    lag_table = pd.concat([lag_table[~lag_table.column.isin(slow_rules.column)], slow_rules], ignore_index=True)
    lag_table.to_csv(os.path.join(DOCS, 'market_data_publication_lags.csv'), index=False, encoding='utf-8-sig')
    display(slow_rules)

    print(f'market table: {market.shape[1]} columns; missing target months in 1980-01..2021-11 (slow block):')
    display(market.loc['1980-01':'2021-11', SLOW_COLS].isna().sum().to_frame('missing months').T)

# %% [markdown]
# ### 6.3 Merge market data (§6.1, §6.2, §6.2b) into the §5 feature table
#
# The input is Cathy's `panel_top500_features.parquet` (§5.8): the same rows as `panel_top500_clean.parquet`, plus the volatility features. One row per (`permno`, `target_month`). `validate='many_to_one'` makes pandas raise if any target month were duplicated in `market`, and the row count must not change.
#
# **`svar` vs `mkt_lvol`.** §5's `mkt_lvol` is $\tfrac12\log(\text{svar}/21)$ of month *t−1*. That is the same information as §6.1's lagged `svar` / `log_svar`, just rescaled. We first **check that the two agree**, which confirms that §5 and §6 use the same timing. Then we keep `mkt_lvol`, because `rel_vol` is built from it, and drop `svar` and `log_svar` from the merged table.

# %%
panel = pd.read_parquet(os.path.join(OUT, 'panel_top500_features.parquet'))      # §5.8 output
panel['target_ym'] = panel['target_month'].dt.to_period('M')

# §5 mkt_lvol and §6.1 lagged svar must describe the same month (both = month t-1)
mkt_lvol_61 = panel['target_ym'].map(0.5 * (market['log_svar'] - np.log(DAYS_PER_MONTH)))
assert same_values(panel['mkt_lvol'], mkt_lvol_61), 'mkt_lvol (§5) and svar (§6.1) are not aligned to the same month'
print('mkt_lvol (§5) matches lagged svar (§6.1) on every row -> keep mkt_lvol, drop svar / log_svar')

n0 = len(panel)
merged = panel.merge(market.drop(columns=['svar', 'log_svar']), left_on='target_ym', right_index=True,
                     how='left', validate='many_to_one')
assert len(merged) == n0, 'merge changed the row count'
MARKET_X = [c for c in market.columns if c not in ('svar', 'log_svar')]          # market inputs for the models
print(f'merged: {merged.shape[0]:,} rows x {merged.shape[1]} cols, {merged.memory_usage(deep=True).sum()/2**20:,.0f} MB; '
      f'{len(MARKET_X)} market columns ({len(SLOW_COLS)} of them the optional slow block)')

# %% [markdown]
# ### 6.4 Leakage audit of the merged table
#
# #### 6.4a Every column: latest month of information, and whether it may be an input
#
# The rule is that a model input for target month *t* may only use information available at the end of month *t−1*. The table below assigns **every column** of `merged` a role and the latest month it uses:
# * stock-panel columns follow the timing verified in §2.4 and GKX's own publication lags (monthly 1 month; quarterly Compustat at least 4; annual at least 6);
# * market columns follow the rules of §6.1–6.2b;
# * §5's features are covered by Cathy's truncation test (§5.2).
#
# The cell **fails if any column of `merged` is missing from the audit**. A new column therefore cannot reach the models without a timing rule. The cell's output is `MODEL_INPUTS`, the list every MS3 model should take its inputs from.

# %%
GKX_FREQ = {'M': ('t-1', 'monthly (CRSP-based); GKX lag 1 month, timing verified in §2.4'),
            'Q': ('quarter ending <= t-5', 'quarterly Compustat; GKX lag >= 4 months'),
            'A': ('fiscal year ending <= t-7', 'annual Compustat; GKX lag >= 6 months')}
rows = [('permno', 'key', '-', 'stock identifier'),
        ('date', 'key', '-', 'GKX row date = month t; its contents describe month t-1 (§2.4)'),
        ('target_month', 'key', '-', 'month t whose volatility is forecast'),
        ('target_ym', 'key', '-', 'target_month as a monthly period (merge key)'),
        ('y', 'target', 't', 'log retvol of month t, read from row t+1 (§5.3)'),
        ('size_rank', 'universe filter', 't-1', 'rank of lagged market equity mvel1 (§3.3)'),
        ('sic2', 'input', 't (industry code; see check below)', 'SIC code on the row date (GKX)'),
        ('sector', 'input', 't (industry code; see check below)', 'SIC division built from sic2 (§3.3; Nic refines in §7)')]
rows += [(c, 'input', 't-1', 'strict HAR lag (§3.2, §5.1)') for c in ['har_1', 'har_3', 'har_12']]
rows += [(c, 'input', 't-1', 'volatility feature (§5.1); passes the truncation test (§5.2)') for c in NEW_X]
rows += [(c, 'evaluation only', 'future', 'look-ahead flag (§5.3, §5.6); never an input') for c in META]
rows += [(c, 'input', GKX_FREQ[dd.loc[c, 'freq']][0], f"{dd.loc[c, 'meaning']}; {GKX_FREQ[dd.loc[c, 'freq']][1]}") for c in CHARS]
lt = lag_table.drop_duplicates('column', keep='last').set_index('column')
MKT_COLS = [c for c in market.columns if c in merged.columns]          # all market columns, incl. ones 6.5 drops
rows += [(c, 'input (optional block)' if c in SLOW_COLS else 'input', lt.loc[c, 'latest month used'], lt.loc[c, 'reason'])
         for c in MKT_COLS]
audit = pd.DataFrame(rows, columns=['column', 'role', 'latest month used', 'source / rule']).set_index('column')

assert audit.index.is_unique, 'a column is listed twice in the audit'
unaudited = [c for c in merged.columns if c not in audit.index]
assert not unaudited, f'columns without a timing rule (add them to the audit): {unaudited}'
audit = audit.loc[merged.columns]                                          # same order as the table

MODEL_INPUTS = audit.index[audit.role.str.startswith('input')].tolist()
assert not set(MODEL_INPUTS) & set(['y'] + LOOK_AHEAD + META), 'a look-ahead column is marked as an input'
assert not audit.loc[MODEL_INPUTS, 'latest month used'].eq('future').any()

# sic2 is dated month t: how often does a stock's industry code change from one month to the next?
s2 = merged.sort_values(['permno', 'target_month'])[['permno', 'sic2']]
prev = s2.groupby('permno')['sic2'].shift()
changed = prev.notna() & s2['sic2'].notna() & s2['sic2'].ne(prev)
print(f'sic2 changes from the previous month in {changed.sum():,} of {prev.notna().sum():,} stock-months ({changed.mean():.3%})')

audit.to_csv(os.path.join(DOCS, 'merged_table_leakage_audit.csv'), encoding='utf-8-sig')
display(audit.groupby(['role', 'latest month used']).size().rename('columns').to_frame())
print(f'{len(MODEL_INPUTS)} model inputs ({sum(audit.role == "input (optional block)")} in the optional slow block); '
      f'full table -> {DOCS}/merged_table_leakage_audit.csv')
with pd.option_context('display.max_rows', 40, 'display.max_colwidth', 80):
    display(audit.drop(CHARS))                                              # the 94 characteristics are in the csv

# %% [markdown]
# **Reading the audit.**
# * `sic2` (and `sector`, built from it) is dated month *t*. A stock's code changes from one month to the next in **8 of 366,653 stock-months (0.002%)**, so this has no practical effect; both stay as inputs. A strict alternative would be to use the previous row's code.
# * `MODEL_INPUTS` has **146 columns** (6 in the optional slow block). It drops to **145** after §6.5 removes `lty`. Every input except `sic2`/`sector` uses information from *t−1* or earlier.

# %% [markdown]
# #### 6.4b Placebo test: can the audit see a leak?
#
# *Plain version.* We leak on purpose and check that it shows. Each market variable is moved **one month forward**, so the forecast for month *t* uses month *t*'s value, which is not known yet. A simple model is fitted with the correct version and with the leaked version. If the timing really matters, the leaked version should forecast noticeably better. That would show that a timing error of one month would be visible, and that our correctly aligned columns don't contain one.
#
# *Technical version.*
# * **Model:** pooled OLS of `y` on the HAR lags (`har_1`, `har_3_f`, `har_12_f`, as in §5.7) plus one market variable at a time, and once with all fast market variables together.
# * **Evaluation:** validation MSE pooled over the five tuning folds (`walk_forward_splits(mode='tune')`), relative to HAR alone on the **same rows**.
# * **Leak:** `market` shifted by one month, so the value at target month *t* is the one that is correct for *t+1*.
# * `svar` enters as `log_svar`.
#
# **What to expect.** Fast market-volatility series (`log_svar`, the FF factor returns and volatilities) should show a large leak gain, because month-*t* market turbulence moves every stock's month-*t* volatility. Slow series (quarterly and annual, valuation ratios) barely change in one month, so the placebo cannot detect a leak in them. For those, the rule table in §6.4a is the evidence.

# %%
# walk_forward_splits is defined in §6.0

base = merged[(merged.target_month.dt.year >= 1980) & merged.y.notna()].dropna(subset=HAR_X).reset_index(drop=True)
folds = list(walk_forward_splits(base, mode='tune'))
yv, H = base['y'].to_numpy(float), base[HAR_X].to_numpy(float)
aligned = market.reindex(base['target_ym'])                                # correct lags (§6.1-6.2b)
leaked = market.set_axis(market.index - 1).reindex(base['target_ym'])     # value for t+1 placed at t: uses month-t info

def pooled_val_mse(X, ok):
    """Validation MSE of pooled OLS y ~ X, pooled over the five tuning folds, on rows where ok."""
    se, n = 0.0, 0
    for f in folds:
        tr, ev = f.train_idx[ok[f.train_idx]], f.eval_idx[ok[f.eval_idx]]
        b, _ = _ols(X[tr], yv[tr])
        e = yv[ev] - np.column_stack([np.ones(len(ev)), X[ev]]) @ b
        se += e @ e; n += len(ev)
    return se / n

FAST = ['log_svar'] + [c for c in MKT_COLS if c not in SLOW_COLS + ['impvar', 'vrp', 'lty']]   # full-history, monthly-moving
res = []
for name, cols in [(c, [c]) for c in ['log_svar'] + MKT_COLS] + [('ALL fast market variables', FAST)]:
    a, l = aligned[cols].to_numpy(float), leaked[cols].to_numpy(float)
    ok = np.isfinite(a).all(axis=1) & np.isfinite(l).all(axis=1)
    har = pooled_val_mse(H, ok)
    res.append({'variable': name, 'rows': int(ok.sum()),
                'aligned: MSE / HAR': pooled_val_mse(np.column_stack([H, a]), ok) / har,
                'leaked: MSE / HAR': pooled_val_mse(np.column_stack([H, l]), ok) / har})
placebo = pd.DataFrame(res).set_index('variable')
placebo['leak gain (% of HAR MSE)'] = 100 * (placebo['aligned: MSE / HAR'] - placebo['leaked: MSE / HAR'])
placebo = placebo.sort_values('leak gain (% of HAR MSE)', ascending=False)
placebo.to_csv(os.path.join(DOCS, 'placebo_leakage_test.csv'), encoding='utf-8-sig')
display(placebo.round(4))

top = placebo.head(12).iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4.2))
yy = np.arange(len(top))
ax.barh(yy + 0.2, 100 * (1 - top['aligned: MSE / HAR']), height=0.4, color='C0', label='correctly aligned (t-1)')
ax.barh(yy - 0.2, 100 * (1 - top['leaked: MSE / HAR']), height=0.4, color='C3', label='leaked (uses month t)')
ax.set_yticks(yy); ax.set_yticklabels(top.index); ax.axvline(0, color='k', lw=0.8)
ax.set(title='Placebo test: validation MSE reduction vs HAR alone (2011-2015)', xlabel='% lower MSE than HAR')
ax.legend(loc='lower right'); plt.tight_layout(); plt.show()
allr = placebo.loc['ALL fast market variables']
print(f"All fast market variables: aligned {100 * (1 - allr['aligned: MSE / HAR']):.1f}% lower MSE than HAR, "
      f"leaked {100 * (1 - allr['leaked: MSE / HAR']):.1f}% lower -> a one-month timing error {'IS' if allr['leak gain (% of HAR MSE)'] > 1 else 'is NOT'} clearly visible")

# %% [markdown]
# **Reading the placebo test** (validation 2011–2015, 250,673 stock-months).
# * **The test can see a one-month leak.** With all 25 fast market variables leaked, validation MSE drops to **0.776 × HAR** (22.4% lower). Correctly aligned, the same variables give **1.044 × HAR**. The gap is 26.9% of HAR's MSE.
#   * `log_svar` alone: 0.807 leaked vs 1.000 aligned (a gap of 19.3 points).
#   * `dfr` (corporate minus government bond return): 7.5 points.
#   * `ff_mkt` 2.8, `impvar` 2.6, `vrp` 2.0, `ff_mkt_vol3` 1.3.
# * **Slow series show no gain** (under 0.5 points): valuation ratios, interest-rate levels, and the quarterly/annual block. A one-month shift barely changes them, so for these columns the evidence is the rule table in §6.4a.
# * **A second finding, for MS3.** Correctly aligned, the market variables add **almost nothing to a linear pooled HAR**.
#   * The best single variable is `ff_mkt` (0.984 × HAR, 1.6% lower MSE). Most are within ±0.5% of HAR.
#   * All fast variables together are **4.4% worse** than HAR alone.
#   * The likely reasons: last month's market volatility is already inside every stock's `har_1`, and a linear fit with 25 correlated market inputs adapts to the training period. So any gain from market data will have to come from regularization, nonlinear models (interactions with the stock's own volatility), or stress months, not from adding the columns to a linear HAR.

# %% [markdown]
# ### 6.5 Scaling and missingness of the market columns
#
# The report has one row per market column. It uses **one value per month**, which is the natural unit for market data: every stock in a month gets the same value. It shows:
# * coverage: first and last target month, and missing months in the train (1980–2010), validation (2011–2015) and test (2016–2021-11) windows;
# * the training-window distribution (min, max, mean, std);
# * **drift**: the test-window mean in training z-units, and the share of test months **outside the training range**. Tree models cannot extrapolate: a test value beyond the training range lands in the same leaf as the most extreme training value. Linear models do extrapolate, so a trending input can push their forecasts off.
#
# **Scaling rule (§3.6).** HAR lags, §5's volatility features and market columns are z-scored with the mean and std of **each fold's training rows**, refit at every walk-forward step. This uses `fold_zscore()` (§6.0). Missing values become 0, which equals the training mean, and get a 0/1 `<col>_missing` indicator. Characteristics keep their monthly rank transform to [−1, 1] (§3.6), which needs no fitting. `sic2`/`sector` are categorical, and `flag_first` is already 0/1.

# %%
# fold_zscore is defined in §6.0
BLOCK = {**{c: 'Goyal-Welch monthly' for c in list(GW_RULES) + ['e/p']},
         **{c: 'Fama-French 5' for c in ff_feat.columns}, **{c: 'GW quarterly/annual (optional)' for c in SLOW_COLS}}
WIN = {'train': ('1980-01', '2010-12'), 'validation': ('2011-01', '2015-12'), 'test': ('2016-01', '2021-11')}
M = market[MKT_COLS]
tr, te = M.loc[slice(*WIN['train'])], M.loc[slice(*WIN['test'])]
rep = pd.DataFrame({'block': [BLOCK[c] for c in MKT_COLS],
                    'first target month': [str(M[c].first_valid_index()) for c in MKT_COLS],
                    'last target month': [str(M[c].last_valid_index()) for c in MKT_COLS]}, index=pd.Index(MKT_COLS, name='column'))
for k, (a, b) in WIN.items():
    rep[f'missing months, {k}'] = M.loc[a:b].isna().sum()
rep['train min'], rep['train max'], rep['train mean'], rep['train std'] = tr.min(), tr.max(), tr.mean(), tr.std()
rep['test mean (train z-units)'] = (te.mean() - tr.mean()) / tr.std()
rep['test months outside train range %'] = 100 * ((te.lt(tr.min()) | te.gt(tr.max())) & te.notna()).sum() / te.notna().sum()
rep.to_csv(os.path.join(DOCS, 'market_scaling_missingness_report.csv'), encoding='utf-8-sig')
with pd.option_context('display.max_rows', 50):
    display(rep.sort_values('test months outside train range %', ascending=False).round(3))
print('start after 1980-01 (missing early in training):', rep.index[rep['missing months, train'] > 0].tolist())
print('end before 2021-11:', rep.index[rep['missing months, test'] > 0].tolist() or 'none')

# lty = tms + tbl exactly, and it trends out of the training range: drop it from the inputs (keep tbl and tms)
both = market[['lty', 'tms', 'tbl']].dropna()
assert np.allclose(both['lty'], both['tms'] + both['tbl'], atol=1e-4), 'lty is no longer tms + tbl'
DROP_INPUTS = ['lty']
MODEL_INPUTS = [c for c in MODEL_INPUTS if c not in DROP_INPUTS]
MARKET_X = [c for c in MARKET_X if c not in DROP_INPUTS]
audit.loc[DROP_INPUTS, 'role'] = 'dropped (6.5: = tms + tbl, out of training range)'
audit.to_csv(os.path.join(DOCS, 'merged_table_leakage_audit.csv'), encoding='utf-8-sig')
print(f'dropped {DROP_INPUTS}; {len(MODEL_INPUTS)} model inputs remain')

# per-fold z-scores: fitted on each fold's training rows only
ZSCORE_COLS = [c for c in MODEL_INPUTS if c not in CHARS + ['sic2', 'sector', 'flag_first']]
stats = {}
for f in [folds[0], list(walk_forward_splits(base, mode='test'))[-1]]:
    Z, st = fold_zscore(base, ZSCORE_COLS, f.train_idx)
    zt = Z.iloc[f.train_idx][ZSCORE_COLS]
    assert np.allclose(zt.mean(), 0, atol=1e-6), 'training rows must have mean 0 after scaling'
    stats[f.name] = st
drift = pd.concat(stats, axis=1).loc[['tbl', 'tms', 'dfy', 'mkt_lvol', 'har_1', 'ff_mkt_vol3']]
print(f'z-score parameters differ by fold (first tuning fold vs last test refit), so they must be refit each time; '
      f'{Z.shape[1] - len(ZSCORE_COLS)} missing-indicator columns:')
display(drift.round(4))

# %% [markdown]
# **Reading the report.**
# * **Late starts.** `impvar` (from 1996), `vrp` (from 1990) and `crdstd` (from September 1990) are missing for 193, 121 and 128 of the 372 training months. Nothing ends before 2021-11. Their `_missing` indicators let linear models use them anyway; whether they are worth keeping is a validation question for MS3.
# * **Drift** (test-window mean in training z-units / share of test months outside the training range):
#   * `lty`: −2.0 / 96%. Long yields fell below anything seen in 1980–2010. It is also exactly `tms + tbl`, so it is **dropped**.
#   * `cfacc`: +2.1 / 17%. It is in the optional slow block, which is one more reason that block has to earn its place on validation.
#   * `impvar`: −0.9 / 13%. Implied variance in 2016–2021 often sat below its 1996–2010 lows.
#   * `govik` −1.6, `tbl` −1.3, `ntis` −1.0: shifted, but almost always inside the training range (0–3% outside). Trees see familiar values; linear models see unusually large z-scores.
#   * The Fama–French columns barely move (|z| ≤ 0.34, at most 3% outside), as expected for returns and their volatilities.
# * **Why per-fold scaling.** The z-score parameters change between folds: for example, the mean of `tbl` is 5.3% in the first tuning fold and 4.2% for the last test refit. `fold_zscore()` refits them every time.

# %% [markdown]
# ### 6.7 Walk-forward splits *(shared by every model; functions defined in §6.0)*
#
# An expanding window throughout: every training set starts in 1980-01 and grows by one year per fold. A **1-month embargo** separates the last training target month from the first evaluation month.
#
# * `walk_forward_splits(df, mode='tune')`: five folds that validate on 2011, …, 2015. **Choose hyperparameters here, then freeze them.**
# * `walk_forward_splits(df, mode='test')`: **annual refit**. Six folds; for each test year 2016, …, 2021, retrain through November of the year before and predict that year. The test window ends 2021-11. Test folds re-estimate coefficients only and never change model choices.
#
# Every model, the per-stock HAR baseline included, uses the same folds, so loss comparisons are like for like. Each fold returns positional row indices (`f.train_idx`, `f.eval_idx`, for `.iloc`), so the row order of `df` doesn't matter. Anything fitted from data (z-scores, imputation medians) must be fitted on `f.train_idx` inside each fold, never once on the whole table.
#
# ```python
# # walk_forward_splits is defined in §6.0 of this notebook
# for f in walk_forward_splits(merged, mode='tune'):
#     model.fit(X.iloc[f.train_idx], y.iloc[f.train_idx])
#     val_loss = loss(y.iloc[f.eval_idx], model.predict(X.iloc[f.eval_idx]))
# ```

# %%
# walk_forward_splits and describe_splits are defined in §6.0

# the split definitions (§6.0) must agree with the dates fixed in the setup cell
assert TUNE_YEARS[0] - 1 == TRAIN_END.year and TUNE_YEARS[-1] == VAL_END.year

fold_table = describe_splits(merged)
fold_table.to_csv(os.path.join(DOCS, 'walk_forward_folds.csv'), encoding='utf-8-sig')
display(fold_table)

# %% [markdown]
# ## 7. Sector labels, data dictionary, staff caveats, revised proposal & slides  *(owner: **Nic**)*
#
# This section adds **no new code**. Every item is either already computed earlier in the notebook or is a separate deliverable, so nothing here needs to be rerun.
#
# | Item | Status | Where |
# |---|---|---|
# | Sector labels | **Kept as the SIC-division labels built in §3.3** (`sector`, 11 groups), plus the raw 2-digit code `sic2`. The Fama–French 12-industry mapping is **deferred to MS3** as an optional refinement. | §3.3; used in §3.4, §4.6, §4.7 |
# | Data dictionary and summary statistics | Done for every GKX, Goyal–Welch and Fama–French column, saved as CSV. The merged table's column-by-column timing audit is also saved. | §2.1–2.2, §6.4a; folder `Data Definition and Summary Statistics/` |
# | Staff caveats (project *Approved*) | Portfolio / Sharpe backtest dropped. Regularized regression and gradient-boosted trees are the main models; clustering (§4.6) and neural networks stay optional. yfinance is no longer used: GKX keeps delisted stocks, so there is no survivorship bias. SEC XBRL data are not used. | §8; revised proposal §1 and §4; slides 11–12 |
# | Revised proposal (2 pages) | Done | `Group_7_Proposal_Revised.pdf` |
# | Slides (14, numbered) | Done | `Group 7 Milestone 2 Slides v2.pdf` |
#
# **Why the Fama–French 12 mapping is deferred.**
# * *Plain version.* We already sort every company into one of 11 broad industry groups (manufacturing, finance, utilities, and so on), and §4.7 shows these groups already separate calm and volatile stocks. A finer, finance-specific grouping would mainly help the models in the next milestone, and it does not change the cleaned table, so it can wait.
# * *Technical version.* FF12 needs 4-digit SIC ranges (e.g. 2830–2836 for Healthcare), but GKX gives only `sic2`, so an FF12 mapping from `sic2` would be approximate. Changing `sector` would also change §3.3 onward: the sector tables in §3.4, the clustering cross-tab in §4.6, the boxplots in §4.7 and the saved parquet files. That means a full rerun. Models can already use `sic2` directly (one-hot), so no information is lost by waiting. If MS3 adds FF12, it goes in the §3.3 cell, followed by a top-to-bottom rerun.
#

# %% [markdown]
# ## 8. Revised project question
#
# **Main question (forecasting, regression).**
# * **Setting:** the 500 largest U.S. stocks each month, 1980–2021, with information available at the end of month *t−1*.
# * **Question:** do regularized regression (ridge, lasso, elastic net) and gradient-boosted trees, using HAR lags, volatility features, firm characteristics and market data, forecast a stock's **log realized volatility in month *t*** better than **pooled HAR(1, 3, 12-month)**?
# * **Test:** out of sample on target months 2016-01 to 2021-11, measured by MSE on log volatility and QLIKE on variance. Significance is assessed at the 5% level with a Diebold–Mariano test on the monthly cross-sectional average loss differential.
#
# **Secondary questions.**
# 1. Which features account for any improvement (permutation importance, lasso selection stability across bootstraps)?
# 2. Does the improvement hold separately in calm months and in stress months, where stress means `svar` is above its training-period 90th percentile? We report block-bootstrap confidence intervals, plus a secondary 2008–09 out-of-sample check.
#
# **How we will know it was answered:** the DM-test p-value and the loss ratio (model ÷ pooled HAR) on the fixed test set, reported for every model we try. Per-stock HAR and EWMA are reported alongside for reference.
#
# #### Why the benchmark is pooled HAR, not per-stock HAR
#
# The numbers on the slides (slide 10: 0.1256 / 0.1306 / 0.1473) come from the §5.7 baseline check on the 2011–2015 validation years, 29,943 stock-months scored by all three baselines:
#
# | Baseline | What is estimated | Validation MSE (log vol) | QLIKE |
# |---|---|---|---|
# | **Pooled HAR** | one set of 4 coefficients shared by all stocks | **0.1256** | **0.274** |
# | Per-stock HAR | 4 coefficients for each stock with ≥ 36 training months (92% of rows); pooled HAR for the rest | 0.1306 | 0.277 |
# | EWMA (λ = 0.5) | nothing | 0.1473 | 0.321 |
#
# Pooled HAR beats per-stock HAR by 0.0050 in MSE (Diebold–Mariano t = −2.50, p = 0.013). The result is not caused by the cold-start rule: per-stock HAR is still worse when it requires 12, 36 or 60 months of history (0.1325, 0.1306, 0.1290).
#
# *Plain version.* Imagine guessing each student's next exam score from their past scores. You could fit a separate rule for every student, but a student with only a few past exams gives you a shaky rule. Or you could learn one rule from all students together ("last score matters this much, the term average this much"), which is far more stable. Here, many stocks have only a few years of monthly history, so a separate rule per stock is shaky, and the shared rule forecasts better. We want our machine-learning models to be compared with the **strongest** simple method, not a weak one. Otherwise "ML beats HAR" could just mean "anything beats a noisy benchmark".
#
# *Technical version.*
# * **Bias–variance.** Per-stock OLS estimates 4 parameters from each stock's own monthly history (36 months at minimum; the median stock spends only 57 months in the universe, §3.3), and the HAR regressors are highly collinear (`har_1`, `har_3`, `har_12` overlap by construction). The coefficient noise (variance) costs more than the stock-specific slopes gain (bias). Pooling imposes common slopes, which is the extreme of the partial pooling in L17 (hierarchical models).
# * **Fair comparison.** Every ML challenger here is itself pooled, so pooled HAR isolates the question we care about: does *extra information or nonlinearity* help, holding the pooling fixed?
# * **Stock-specific levels are not lost.** The HAR lags already carry each stock's own volatility level, so pooled HAR still forecasts a calm utility lower than a volatile tech stock. What it shares is only how quickly volatility reverts.
# * **Caveat.** These validation numbers come from one fit on 1980–2010 targets, without the one-month embargo of the final protocol (§6.0, §6.7). In MS3 all baselines are rerun on the walk-forward folds before any ML comparison.
#
# *Changed from the original proposal:*
# * the horizon moves from week to month and the data from daily yfinance to monthly GKX, because no daily OHLC data are available;
# * the benchmark moves from per-stock HAR to **pooled HAR**, because pooled HAR forecast better on validation (above);
# * the test window moves from 2019–2026 to 2016–2021, because the data end in 2021;
# * the economic-value backtest is dropped, as the staff required.
#
