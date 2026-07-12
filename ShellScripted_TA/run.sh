#!/bin/bash

HEADER_FILE="TsetlinMachine.h"
C_FILE="TsetlinMachine.c"


# Dataset files (Modify this as needed)
X_TRAIN_FILE="Booleanized_Data/x_train.npy"
X_TEST_FILE="Booleanized_Data/x_test.npy"



# === Step 1: Ask user for all parameters ===
echo "Enter X_Train path:"
read X_TRAIN_FILE

echo "Enter Y_Train path:"
read Y_TRAIN_FILE

echo "Enter X_Test path:"
read X_TEST_FILE

echo "Enter Y_test path:"
read Y_TEST_FILE

echo "Enter CLAUSES:"
read CLAUSES

echo "Enter CLASSES:"
read CLASSES

echo "Enter THRESHOLD:"
read THRESHOLD

echo "Enter SPECIFICITY:"
read S

echo "Enter EPOCHS:"
read EPOCHS

# Validate input
if [ -z "$X_TRAIN_FILE" ] || [ -z "$Y_TRAIN_FILE" ] || [ -z "$X_TEST_FILE" ] || [ -z "$Y_TEST_FILE" ]; then
    echo "[ERROR] Missing dataset paths."
    exit 1
fi

echo "[INFO] Detecting dataset parameters from $X_TRAIN_FILE and $X_TEST_FILE..."

NUMBER_OF_EXAMPLES_TRAIN=$(python3 -c "import numpy as np; print(np.load('$X_TRAIN_FILE', allow_pickle=True).shape[0])")
NUMBER_OF_EXAMPLES_TEST=$(python3 -c "import numpy as np; print(np.load('$X_TEST_FILE', allow_pickle=True).shape[0])")
FEATURES=$(python3 -c "import numpy as np; print(np.load('$X_TRAIN_FILE', allow_pickle=True).shape[1])")

echo "[INFO] TRAIN=$NUMBER_OF_EXAMPLES_TRAIN, TEST=$NUMBER_OF_EXAMPLES_TEST, FEATURES=$FEATURES"
# === Create unique run directory ===
RUN_NAME="CL${CLAUSES}_T${THRESHOLD}_S${S}_E${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/$RUN_NAME"

mkdir -p "$RUN_DIR"
echo "[INFO] Run directory: $RUN_DIR"

# === Step 2: Modify the .h file dynamically ===
sed -i "s/^#define CLAUSES .*/#define CLAUSES $CLAUSES/" $HEADER_FILE
sed -i "s/^#define CLASSES .*/#define CLASSES $CLASSES/" $HEADER_FILE
sed -i "s/^#define NUMBER_OF_EXAMPLES_TRAIN .*/#define NUMBER_OF_EXAMPLES_TRAIN $NUMBER_OF_EXAMPLES_TRAIN/" $HEADER_FILE
sed -i "s/^#define NUMBER_OF_EXAMPLES_TEST .*/#define NUMBER_OF_EXAMPLES_TEST $NUMBER_OF_EXAMPLES_TEST/" $HEADER_FILE
sed -i "s/^#define THRESHOLD .*/#define THRESHOLD $THRESHOLD/" $HEADER_FILE
sed -i "s/^#define FEATURES .*/#define FEATURES $FEATURES/" $HEADER_FILE

echo "[INFO] Updated $HEADER_FILE with your parameters."

# === Step 3: Compile the C program ===
echo "[INFO] Compiling $C_FILE..."
gcc -O3 -ffast-math -shared -o tsetlin.so -fPIC TsetlinMachine.c
if [ $? -ne 0 ]; then
    echo "[ERROR] Compilation failed."
    exit 1
fi
PY_SCRIPT="python3 -u TM_run.py \
    --X_train $X_TRAIN_FILE \
    --Y_train $Y_TRAIN_FILE \
    --X_test $X_TEST_FILE \
    --Y_test $Y_TEST_FILE \
    --features $FEATURES \
    --num_classes $CLASSES \
    --s $S \
    --num_epochs $EPOCHS \
    --clauses $CLAUSES \
    --save_dir $RUN_DIR/Model_Saved \
    --ta_dir $RUN_DIR/TA_STATE \
    --cls_dir $RUN_DIR/Class_Sums \
    --result_file $RUN_DIR/Result.txt"
# === Step 4: Run Python script with arguments ===
echo "[INFO] Running Python script..."
$PY_SCRIPT
if [ $? -ne 0 ]; then
    echo "[ERROR] Python script execution failed."
    exit 1
fi

# === Step 5: Capture accuracy from Python script ===
# Modify TM_run.py to print accuracy in a structured way like:
# ACCURACY: 92.34



# Directory where TM_run.py saved the raw TA states:
TA_DIR="$RUN_DIR/TA_STATE"
mkdir -p "$TA_DIR"

# Directory where compressed TA will be stored:
COMPRESSED_DIR="$RUN_DIR/COMPRESSED_TA"
mkdir -p "$COMPRESSED_DIR"


# TA filename pattern (modify if needed)
TA_FILE="$RUN_DIR/TA_STATE/TA_${FEATURES}_${CLAUSES}.txt"

# Validate TA file exists
if [ ! -f "$TA_FILE" ]; then
    echo "[ERROR] TA states file not found at: $TA_FILE"
    echo "[HINT] Check your TM_run.py TA saving path."
    exit 1
fi

echo "[INFO] Found TA states file: $TA_FILE"
echo "[INFO] Running TA Compression..."
python3 TA_Compression.py \
    "${dataset_name:-TM_Data}" \
    "$THRESHOLD" \
    "$CLAUSES" \
    "$CLASSES" \
    "$FEATURES" \
    "$TA_FILE" \
    --write-datastore \
    --outdir "$COMPRESSED_DIR"

if [ $? -ne 0 ]; then
    echo "[ERROR] TA Compression failed."
    exit 1
fi

echo "[INFO] Training summary saved to $RUN_DIR/Result.txt"

echo "[INFO] Completed successfully. Model and predictions saved."
