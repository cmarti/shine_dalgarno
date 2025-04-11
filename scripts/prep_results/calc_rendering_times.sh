echo "conda activate gpmap & export PYTHONPATH="$PYTHONPATH:`pwd`" & python calc_rendering_times.py" | qsub -N times -l mem_free=32G -pe threads 8 -cwd
