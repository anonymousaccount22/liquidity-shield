# Fechamento Letter — Bloco A

Painéis existentes: `painel_regressao_q_bp.csv` e `painel_regressao_q_dfc_ortog.csv`. A função `serie_y` foi carregada isoladamente de `projecoes_locais.py` para evitar executar o loop de estimações.

## A1. Trimestres
| Amostra/h | T usados | Primeiro | Último |
|---|---|---|---|
| Painel atual | 55 | 2011Q1 | 2024Q3 |
| Cheia, h=3 | 54 | 2011Q2 | 2024Q3 |
| Cheia, h=5 | 54 | 2011Q2 | 2024Q3 |
| Cheia, h=7 | 54 | 2011Q2 | 2024Q3 |
| Restrita, h=3 | 54 | 2011Q2 | 2024Q3 |
| Restrita, h=5 | 54 | 2011Q2 | 2024Q3 |
| Restrita, h=7 | 54 | 2011Q2 | 2024Q3 |

Trimestres de choque não usados: 2011Q1. Primeiro trimestre com choque cai por `Shock_L1` ausente; h não muda o conjunto de trimestres.

## A2. Concentração na amostra final
Base: `(Shock_t - média)^2` sobre os trimestres efetivamente usados em cada amostra/horizonte.
| Amostra | h | T | Share 2021 | Share 2021Q4 | N efetivo |
|---|---|---|---|---|---|
| Cheia | 3 | 54 | 0,344599 | 0,276828 | 9,7968 |
| Cheia | 5 | 54 | 0,344599 | 0,276828 | 9,7968 |
| Cheia | 7 | 54 | 0,344599 | 0,276828 | 9,7968 |
| Restrita | 3 | 54 | 0,344599 | 0,276828 | 9,7968 |
| Restrita | 5 | 54 | 0,344599 | 0,276828 | 9,7968 |
| Restrita | 7 | 54 | 0,344599 | 0,276828 | 9,7968 |

## A3. Attrition
| Etapa | Firmas | Firma-trimestres | Observação |
|---|---|---|---|
| Painel contábil bruto; pré-filtros institucionais | NÃO RODADO | NÃO RODADO | Painel intermediário pré-filtros não está salvo; não reli toda a pasta CVM. |
| Sem financeiras | NÃO RODADO | NÃO RODADO | Sem output por etapa. |
| Sem estatais | NÃO RODADO | NÃO RODADO | Sem output por etapa. |
| Sem utilities | NÃO RODADO | NÃO RODADO | Sem output por etapa. |
| Ativo fixo válido no painel salvo | 555 | 18199 | Contagem no painel de regressão já filtrado. |
| Amostra cheia no painel salvo | 568 | 18484 | painel_regressao_q_bp.csv |
| Com Tobin's Q defasado | 286 | 10728 | Critério de amostra Restrita; firma-trimestres disponíveis no painel salvo. |
| Amostra restrita reportada | 286 | NÃO RODADO | 286 firmas consta em carry_00_contagem_N.csv; N de linhas dessa etapa não salvo. |
| Usada na estimação, Cheia, h=3 | 476 | 14029 | T=54 |
| Usada na estimação, Restrita, h=3 | 268 | 8920 | T=54 |
| Usada na estimação, Cheia, h=5 | 460 | 13720 | T=54 |
| Usada na estimação, Restrita, h=5 | 268 | 8817 | T=54 |
| Usada na estimação, Cheia, h=7 | 448 | 13406 | T=54 |
| Usada na estimação, Restrita, h=7 | 268 | 8692 | T=54 |

Motivo entre as firmas restritas que não chegam à amostra estimada:
| h | Fora | Sem desfecho | Sem regressor | Sem setor | Sobrepostas |
|---|---|---|---|---|---|
| 3 | 18 | 6 | 12 | 0 | 0 |
| 5 | 18 | 7 | 11 | 0 | 0 |
| 7 | 18 | 11 | 7 | 0 | 0 |

Exclusão de firmas por ano fiscal não calendário: NÃO EXISTE no script_painel.py.

## A4. CAPEX disponível em t..t+h
| Amostra | h | Subconjunto | Firmas | Observações |
|---|---|---|---|---|
| Cheia | 3 | Amostra completa | 210 | 5740 |
| Cheia | 3 | Tercil superior de Tamanho | 58 | 2122 |
| Cheia | 5 | Amostra completa | 208 | 5351 |
| Cheia | 5 | Tercil superior de Tamanho | 55 | 1918 |
| Cheia | 7 | Amostra completa | 203 | 4988 |
| Cheia | 7 | Tercil superior de Tamanho | 54 | 1837 |
| Restrita | 3 | Amostra completa | 210 | 5651 |
| Restrita | 3 | Tercil superior de Tamanho | 54 | 1924 |
| Restrita | 5 | Amostra completa | 207 | 5267 |
| Restrita | 5 | Tercil superior de Tamanho | 53 | 1798 |
| Restrita | 7 | Amostra completa | 203 | 4911 |
| Restrita | 7 | Tercil superior de Tamanho | 51 | 1682 |
Tercil conforme inv1315_desfechos.py: média de Tamanho por firma em 2012Q1–2013Q1; corte de 2/3 dentro da amostra.

## A5. Entrantes em 2021
| Grupo | Firmas únicas | Firma-trimestres observados | Cash_Ratio médio |
|---|---|---|---|
| Entrantes | 123 | 458 | 0,198687 |
| Não entrantes | 240 | 937 | 0,142711 |
Entrante = primeira observação real em 2019Q1–2021Q4, excluídas firmas censuradas à esquerda; média de Cash_Ratio por firma-trimestre em 2021.

## Bloco B
NÃO RODADO: execução limitada ao Bloco A conforme instrução.
