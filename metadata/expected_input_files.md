# Expected Restricted Input Files

The public repository does not include restricted ELSA or HRS files. A reproducing user must obtain them directly from the original data providers.

## ELSA

The canonical manuscript branch starts from already imputed ELSA core wave files for waves 1--9 under the `MNAR_Pmm` branch. The expected local structure is:

```text
<elsa_imputed_wave_dir>/
  w11.csv
  w22.csv
  w33.csv
  w44.csv
  w55.csv
  w66.csv
  w77.csv
  w88.csv
  w99.csv
```

## HRS

The HRS shared-subset workflow starts from a preprocessed Markov-style HRS/RAND table:

```text
<hrs_preprocessed_markov_csv>
```

The paper does not redistribute this file. The public code records the shared ELSA-HRS schema and expects the user to map authorized local HRS/RAND files into the same columns.
