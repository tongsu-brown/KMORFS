@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
set MPLBACKEND=Agg
D:\anaconda3\envs\data2060new\python.exe -u D:\KMORFS_local\KMORFS-db\early_state_stress_thickness\fit_early_state_stress.py > D:\KMORFS_local\KMORFS-db\fit_output.txt 2>&1
