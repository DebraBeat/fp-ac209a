# fp-ac209a — Forecasting Stock Volatility on the S&P 500: Can ML Beat HAR?

CS1090A / AC209A final project, **Group 7** (Project #12).
Members: Debra Ritter · YiTong Li · Yicheng (Cathy) Liu · Anthony Sciola · Nic Bohm.

One row of our table is *one large U.S. stock in one month*; from everything known at the end of
month *t−1* we predict the log realized volatility of month *t*, and ask whether pooled ML models
beat a pooled HAR benchmark.

## Repo layout

```
Notebook/
  MS2_notebook.ipynb   <- Milestone 2 notebook (rendered outputs, what we submit)
  MS2_notebook.py      <- the SAME notebook as a plain script (jupytext "percent" format) -> review/diff this
  splits.py            <- shared walk-forward splits + fold-wise z-scoring (import from any model notebook)
deliverables/          <- slides / PDFs we hand in
Data/                  <- raw data (NOT in git, see Data/README.md)
data_clean/            <- tables the notebooks write (NOT in git)
jupytext.toml          <- keeps every .ipynb paired with its .py twin
requirements.txt
```

## Setup (local)

```bash
git clone https://github.com/DebraBeat/fp-ac209a.git
cd fp-ac209a
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then put the raw data in `Data/` (see `Data/README.md` — the files live in the shared Google Drive
folder `final_project`). Start Jupyter from the repo root (`jupyter lab`) and open `Notebook/MS2_notebook.ipynb`.
The setup cell finds `Data/` whether the kernel's working directory is the repo root or `Notebook/`.

**Colab** still works unchanged: the setup cell mounts Drive and reads from
`MyDrive/final_project` (add Debbie's shared folder as a shortcut to *My Drive* first).

## Notebooks and `.py` twins (jupytext)

Every notebook has a `.py` twin next to it. The `.py` is what GitHub can diff and review; the `.ipynb`
keeps the rendered outputs. They are *the same file in two formats* — never edit only one of them:

* **In Jupyter** (with `jupytext` installed, which `requirements.txt` does): saving the notebook rewrites the
  `.py`; opening a notebook whose `.py` is newer updates it. Nothing else to do.
* **Edited the `.py` in an editor / pulled a change from git?** Run `jupytext --sync Notebook/MS2_notebook.ipynb`
  (updates whichever side is older, outputs are kept).
* **Colab**: Colab does not know about the pairing. After downloading the notebook from Colab, drop it over
  `Notebook/MS2_notebook.ipynb` and run `jupytext --sync Notebook/MS2_notebook.ipynb` before committing.

`jupytext.toml` strips Colab's per-run cell metadata (`executionInfo`, `outputId`, …), so the `.py` only
changes when code or text changes.

Tip: `git diff` on the `.py` is readable; `git diff` on the `.ipynb` is not (outputs + base64 images).
When you review a pull request, look at the `.py`.

## How we work in git

* `main` always holds the latest *merged* notebook that runs end to end.
* Work on a branch named `<you>/<what>` (e.g. `cathy/ms3-har-baseline`), push it, open a pull request
  against `main`, get one other member to look at the `.py` diff, merge.
* Keep `Data/` and `data_clean/` out of git (already in `.gitignore`); share data through Drive.
* Milestone 2 history: each member's draft is kept on `yitong/ms2`, `cathy/ms2`, `anthony/ms2`, `nic/ms2`
  (`nic/ms2` is the merged submission) — `git diff main...cathy/ms2 -- Notebook/MS2_notebook.py` shows what a
  draft added relative to the base.

Useful commands:

```bash
git fetch --all                               # see everyone's branches
git switch -c debbie/ms3-ridge main           # start new work from main
jupytext --sync Notebook/*.ipynb              # make sure .ipynb and .py agree before committing
git diff main -- Notebook/MS2_notebook.py     # what did I change?
```
