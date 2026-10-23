# Replication code: pooled cash gradients, monetary shocks and fixed-asset outcomes in Brazil

This repository contains the code and the constructed estimation panel used in the letter *"Is the liquidity shield a stable channel? Episode dependence in panel local projections with aggregate monetary shocks"*.

The study estimates panel local projections (LPs) in which an aggregate monetary shock, built from the 360-day DI-Pré swap around COPOM meetings, is interacted with the lagged cash ratio of listed non-financial Brazilian firms (2011Q1-2024Q3). The outcome is the change in log fixed assets. The results are conditional panel regressions and carry no causal interpretation.

## 1. What is (and is not) in this repository

**Included**

- The Python scripts that build the panel, construct the shock and run every analysis reported in the paper.
- Small input files: COPOM dates, the daily 360-day DI-Pré series and the monthly EPU series.
- The constructed regression panel (`painel_regressao_q_bp.csv`, about 10 MB) and the saved result files that feed the tables.
- The execution log of the main pipeline.

**Not included**

- The raw CVM files (about 14.6 GB). They are public and must be downloaded from the source (Section 3).
- The raw B3 price archives (`dados_b3/COTAHIST_A*.ZIP`). They are public and must be downloaded from the source.

## 2. Repository structure

```
.
├── main.py                      # runs the main pipeline in order
├── script_painel.py             # builds the accounting panel from CVM files
├── calcular_q.py                # approximate Tobin's Q (CVM equity data + B3 prices)
├── ortogonalizacao.py           # orthogonalises the quarterly shock (VIX, EMBI+)
├── capex_bp.py                  # cash-flow-statement capex (DFC) and balance-sheet variables
├── projecoes_locais.py          # panel LPs: pooled estimates, horizons, IRFs
├── eventos.py                   # event-quarter labels (episodes)
├── ciclos.py                    # episode LPs (2013Q2-2015Q2, 2021, calm quarters)
├── bootstrap_blocos.py          # block bootstrap over quarters
├── placebos.py                  # shock and cash permutation placebos
├── ipo2.py                      # entrant analyses and random sector-matched exclusions
├── fechamento_b1.py             # episode exclusions: baseline vs strict
├── gerar_tabela1.py             # alternative generator of the episode table
├── f_classifica_reunioes.py     # classifies meetings by market reaction (not used in the primary shock)
├── (other analysis scripts)     # [list here any remaining .py files used for the appendix]
├── datas_copom.csv              # COPOM meeting dates
├── swap_di_pre_360_diario.csv   # daily 360-day DI-Pré swap, annual percent
├── epu_brasil.csv               # monthly Brazilian EPU index
├── painel_regressao_q_bp.csv    # constructed regression panel
├── (result files: *.csv, *.md)  # saved outputs used in the tables
└── pipeline_execution.log       # log of the main pipeline run
```

Adjust this tree to the exact file names in the repository.

## 3. Data sources

All data are public. The raw files are not redistributed here.

| Component | Source | Where to put it / notes |
|---|---|---|
| Quarterly financial statements (BPA, BPP, DRE, DFC; DFP/ITR) and equity data of listed firms | CVM, open-data portal [portal address] | [folder name used by the scripts]. Downloaded in [month/year]. |
| Share prices | B3, `COTAHIST_A*.ZIP` | `dados_b3/`. Downloaded in [month/year]. |
| 360-day DI-Pré swap, COPOM dates | B3 and Central Bank of Brazil [confirm] | `swap_di_pre_360_diario.csv`, `datas_copom.csv` (included). |
| Selic target (SGS 432) and monthly IPCA (SGS 433) | Central Bank of Brazil | Used only for the real-rate construction. |
| EMBI+ Brazil (`JPM366_EMBI366`, called `Risco_Fiscal` in the code) | Ipeadata | Quarterly aggregation. |
| VIX (`^VIX`) | Yahoo Finance via `yfinance` | Daily closes averaged within the quarter. |
| Brazilian economic policy uncertainty index | Global Economic Policy Uncertainty Index | `epu_brasil.csv` (included); used only in exploratory tests. |
| Exchange rate and IBC-Br activity index | Central Bank of Brazil [confirm] | Used only in the appendix tests. |

## 4. Environment

The main pipeline was run with the following versions (taken from `pipeline_execution.log`):

- Python 3.13.16
- NumPy 2.5.4
- pandas 3.0.6
- requests 2.34.2
- statsmodels 0.15.0
- linearmodels 7.0
- matplotlib 3.11.2
- yfinance (version not recorded)

```
pip install numpy==2.5.4 pandas==3.0.6 requests==2.34.2 statsmodels==0.15.0 linearmodels==7.0 matplotlib==3.11.2 yfinance
```

## 5. How to reproduce

1. Download the raw CVM and B3 files (Section 3) and place them where the scripts expect them. [Describe the folder layout here.]
2. Install the dependencies (Section 4).
3. Run the main pipeline:

   ```
   python main.py
   ```

   `main.py` runs, in this order: `script_painel.py` -> `calcular_q.py` -> `script_painel.py` (second pass) -> `calcular_q.py` (second pass) -> `ortogonalizacao.py` -> `capex_bp.py` -> `projecoes_locais.py`.

   Principal outputs: `painel_final.csv`, `painel_final_q.csv`, `painel_regressao_q_ortog.csv`, `painel_regressao_q_bp.csv`, `irf_consolidado_Delta_log_Ativo_Fixo_ctrlL1.csv`.
4. Run the analysis scripts for the episode, bootstrap, placebo, entrant and capex results (`ciclos.py`, `bootstrap_blocos.py`, `placebos.py`, `ipo2.py`, `fechamento_b1.py` and the others listed above).

The panel `painel_regressao_q_bp.csv` is provided so that the regressions can be re-run without downloading the raw files.

## 6. Where the paper's tables come from

| Paper element | Saved files |
|---|---|
| Pooled gradient (Table 1, first row of each panel) | outputs of `projecoes_locais.py` (e.g. `irf_consolidado_Delta_log_Ativo_Fixo_ctrlL1.csv`) |
| Table 1 (episodes, exclusions, differences) | `tabela1.csv`, `ciclos_igualdade.csv`, `fechamento_b1.md` |
| Concentration of the shock, leave-one-out | `fechamento_letter.md`, `placebo_loqo.csv`, `placebo_loyo.csv` |
| Block bootstrap | `bootstrap_blocos.csv`, `ipo2_boot.csv` |
| Table 2 (capex vs. stock) | `inv1315_decomp.csv`, `inv1315_split.csv` |
| Table A.1 and Appendix A.2 | alternative-window and leave-one-meeting files, `inv1315_loqo_janela.csv` |
| Appendix A.3 | `carry_01_split.csv`, `carry_02_sem2021.csv`, `carry_04_setorial.csv`, `eventos_epu.csv` |
| Placebos | `placebo_perm.csv`, `placebo_cash.csv`, `placebo_lead.csv`, `placebo_pand.csv` |

## 6.1 Random seeds

- Shock permutation (1,000 draws) and cash permutation (500 draws): seed 12345 (`placebos.py`).
- Entrant-matched random exclusion (1,000 draws) and entrant bootstrap (1,000 draws): seed 20261 (`ipo2.py`).
- Main regime bootstrap (1,000 draws; blocks of length h+1 and 8): seed 20260 (`bootstrap_blocos.py`).

## 7. Notes and known limitations

- Panel quarters: 2011Q1-2024Q3 (55); estimation uses 54 quarters (2011Q2-2024Q3).
- Fiscal years that differ from the calendar year are not adjusted in the main panel.
- Extraordinary COPOM meetings are neither flagged nor removed in the primary shock.
- The standard deviation of the shock is printed by `projecoes_locais.py` when it runs; it is not stored in a saved output file.
- Calm-quarter observation counts and the standard errors of the episode rows in `tabela1.csv` are derived (see the table notes in the paper).
- No stage-by-stage sample attrition table is saved.
- The raw CVM data may be revised by the source after download; the download date is recorded above.

