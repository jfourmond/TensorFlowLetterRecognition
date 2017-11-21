@echo off

SETLOCAL ENABLEDELAYEDEXPANSION

set layers=%1
set staticDirectory=/tmp/dnn-%layers%-layer-

IF "%1"=="" GOTO :end

echo %layers% LAYERS CAMPAIGN

set a=
for /l %%n in (1, 1, %layers%) do (
    set a[%%n]=1
)

for /L %%H in (1, 1, 100) do (
    :: Construction de la chaîne utilisée dans l'appel au script
    set list_layer=
    for /l %%n in (1, 1, %layers%) do (
        set list_layer=!list_layer! %%H
    )
    :: Construction du nom de répertoire accueillant le modèle
    set "directory=%staticDirectory%%%H"
    :: Lancement du script
    python letter-recognition-dnn.py --model_dir=!directory! --results_file=letter-recognition-dnn-%layers%-layers.csv --hidden_units !list_layer!
)

:end

echo Script %0 terminé

ENDLOCAL