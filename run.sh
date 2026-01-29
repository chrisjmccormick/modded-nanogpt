GPUS=8

# ==========================================================
# Baseline

RUNS=4
for i in {1..$RUNS}; do
  echo "Run $i / $RUNS"
  torchrun --standalone --nproc_per_node=$GPUS train_gpt.py
done

# Save the logs
mkdir -p ./logs/baseline
mv ./logs/*.txt ./logs/baseline/

# Summarize results
python results_reporting.py
# ==========================================================


# ==========================================================
# Extra-Warmup-1
# 
RUNS=4
EXP_NAME="extra-warmup-1"

mkdir -p ./logs/$EXP_NAME
mkdir -p ./logs/$EXP_NAME/summaries/

# For each run:
for i in {1..$RUNS}; do
  echo "Run $i / $RUNS"
  torchrun --standalone --nproc_per_node=$GPUS \
    train_gpt-$EXP_NAME.py

  # Move logs to experiment directory
  mv ./logs/*.txt ./logs/$EXP_NAME/

  # Summarize between runs, store in log file
  python results_reporting.py > ./logs/$EXP_NAME/summaries/summary.txt
done
# ==========================================================


# ==========================================================
# Template

# RUNS=4
# EXP_NAME="template"

# for i in {1..$RUNS}; do
#   echo "Run $i / $RUNS"
#   torchrun --standalone --nproc_per_node=$GPUS \
#     train_gpt-$EXP_NAME.py

#   python results_reporting.py
# done

# mkdir -p ./logs/$EXP_NAME
# mv ./logs/*.txt ./logs/$EXP_NAME/

# # Summarize results
# python results_reporting.py
# # ==========================================================
