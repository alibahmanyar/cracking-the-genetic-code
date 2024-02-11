#!/bin/bash
mkdir -p logs
# Activate the virtual environment
source ../.venv/bin/activate

date >> log.txt

# Find and run all .py scripts in the current directory
for script in $(ls ./*.py | sort)
do
    echo "Running script: $script" | tee -a log.txt
    start_time=$(date +%s)
    python3 "$script" | tee -a ./logs/${script}_log.txt
    end_time=$(date +%s)
    execution_time=$(($end_time - $start_time))
    echo "Execution time of $script : $execution_time seconds" | tee -a log.txt
done

# Deactivate the virtual environment
deactivate
