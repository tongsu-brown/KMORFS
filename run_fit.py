import os, sys, subprocess
env = os.environ.copy()
env['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env['MPLBACKEND'] = 'Agg'
env['PYTHONIOENCODING'] = 'utf-8'
env['PYTHONUNBUFFERED'] = '1'
outfile = open(r'D:\KMORFS_local\KMORFS-db\fit_output.txt', 'w', encoding='utf-8')
proc = subprocess.Popen(
    [sys.executable, '-u', r'D:\KMORFS_local\KMORFS-db\early_state_stress_thickness\fit_early_state_stress.py'],
    cwd=r'D:\KMORFS_local\KMORFS-db\early_state_stress_thickness',
    env=env,
    stdout=outfile, stderr=subprocess.STDOUT
)
proc.wait()
outfile.close()
