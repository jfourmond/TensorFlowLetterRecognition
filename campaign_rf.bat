@echo off

SETLOCAL ENABLEDELAYEDEXPANSION

set resultsFile=letter-recognition-rf.csv

for /L %%T in (1, 1, 250) do (
    python letter-recognition-rf.py --n_trees=%%T --random_state=0 --results_file=!resultsFile!
)

echo Script %0 terminé

ENDLOCAL