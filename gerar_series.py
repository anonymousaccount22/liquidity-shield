# pip install ipeadatapy yfinance pandas
import pandas as pd
import yfinance as yf
import ipeadatapy as ipea

# 1. VIX
print("Verificando VIX...")
vix = yf.download('^VIX', start='2000-01-01', progress=False)

if isinstance(vix.columns, pd.MultiIndex):
    close_vix = vix['Close']['^VIX']
else:
    close_vix = vix['Close']

vix_q = close_vix.resample('QE').mean().reset_index()
vix_q.columns = ['Trimestre', 'VIX']
vix_q['Trimestre'] = pd.to_datetime(vix_q['Trimestre']).dt.to_period('Q').astype(str)
vix_q.to_csv('vix_trimestral.csv', index=False)
print('vix_trimestral.csv criado com sucesso!')

# 2. EMBI+ Risco Brasil via ipeadatapy
print('Baixando Risco Brasil (EMBI+)...')
CODIGO = 'JPM366_EMBI366'

df_ipea = ipea.timeseries(CODIGO)

# A coluna de valor tem o nome da unidade (ex.: "VALUE (Points)"), então pego dinamicamente
col_valor = [c for c in df_ipea.columns if c.upper().startswith('VALUE')][0]

# A data pode estar no índice ou na coluna 'RAW DATE', então trato os dois casos
if isinstance(df_ipea.index, pd.DatetimeIndex):
    datas = df_ipea.index
else:
    datas = pd.to_datetime(df_ipea['RAW DATE'], utc=True).dt.tz_localize(None)

df = pd.DataFrame({
    'data': pd.to_datetime(datas).tz_localize(None) if getattr(datas, 'tz', None) else pd.to_datetime(datas),
    'Risco_Fiscal': pd.to_numeric(df_ipea[col_valor].values, errors='coerce'),
}).dropna()

df['Trimestre'] = df['data'].dt.to_period('Q').astype(str)

rf_q = df.groupby('Trimestre')['Risco_Fiscal'].mean().reset_index()
rf_q.to_csv('risco_fiscal_trimestral.csv', index=False)
print('risco_fiscal_trimestral.csv criado com sucesso!')
print(rf_q.tail())