#!/usr/bin/env python3

import numpy as np
import argparse
import os
import collections
import math
import sys

# Constants for bit positions (same as original script)
POLARITY_BIT_VALUE = 1 << 15    # 32768
CLAUSE_CHANGE_BIT_VALUE = 1 << 14  # 16384
ENCODING_OFFSET_MASK = (1 << 14) - 1  # 14 bits for literal offset (0..16383)


def parse_inc_exc_from_raw_states(raw_states, states_threshold):

    raw_states = np.asarray(raw_states)
    return (raw_states > states_threshold).astype(np.uint8)


def include_encoding(num_literals, num_classes, num_clauses, binary_states,
                     output_filename, include_per_class_filename):

    state_type_count = collections.Counter(np.array(binary_states))
    automata = []
    active_TAs_per_class = np.zeros(num_classes, dtype='H')
    clauses_with_all_zero_TAs = 0

    number_of_tas = len(binary_states)
    # iterate per class -> clause -> literal
    for cls in range(num_classes):
        bit_1_counter_class = 0
        polarity = 0
        clause_change = 0

        for cl in range(num_clauses):
            bit_1_counter_clause = 0
            base_idx = (cls * num_clauses + cl) * num_literals
            flag = 0

            for lit in range(num_literals):
                ta_idx = base_idx + lit
                if ta_idx >= number_of_tas:
                    # Defensive check (shouldn't happen if the input sizes are correct)
                    break
                if binary_states[ta_idx] == 1:
                    flag = 1
                    # address encodes: [clause_change_bit | polarity_bit | literal_offset]
                    addr = (lit & ENCODING_OFFSET_MASK) | polarity | clause_change
                    automata.append(addr)
                    bit_1_counter_class += 1
                    bit_1_counter_clause += 1

            if bit_1_counter_clause == 0:
                clauses_with_all_zero_TAs += 1

            # toggle polarity for next clause
            polarity = POLARITY_BIT_VALUE if polarity == 0 else 0
            # clause_change toggles only if there was at least one include TA in this clause
            if flag == 1:
                clause_change = CLAUSE_CHANGE_BIT_VALUE if clause_change == 0 else 0

        active_TAs_per_class[cls] = bit_1_counter_class

    automata = np.array(automata, dtype='H')

    # Print summary
    print('-----------------------------------------------------------------------', flush=True)
    print('TA COMPRESSION SUMMARY:', flush=True)
    print('No of clauses with no include TAs =', clauses_with_all_zero_TAs, flush=True)
    print('No of include TAs =', state_type_count[1], flush=True)
    print('No of exclude TAs =', state_type_count[0], flush=True)
    print('Total no of TAs =', len(binary_states), flush=True)
    print('No of encoded automata entries =', len(automata), flush=True)
    print('-----------------------------------------------------------------------', flush=True)

    # Save outputs
    np.savetxt(include_per_class_filename, active_TAs_per_class, fmt='%u', delimiter=' ')
    np.savetxt(output_filename, automata, fmt='%u', delimiter=' ')

    print('Wrote:', output_filename, flush=True)
    print('Wrote:', include_per_class_filename, flush=True)

    return automata, active_TAs_per_class, int(state_type_count[1])


def generate_include_encoded_setup_h(dataset_name, threshold, clauses, classes, features, states, num_includes, encoded_filename, include_per_class_filename, write_path='.'):
    """
    Generate IncludeEncodedSetup.h with safe placeholders for test-related fields (no test file required).
    NUMBER_OF_TEST_EXAMPLES and TEST_LENGTH_PER32 are set to 0 by default.
    """
    header_path = os.path.join(write_path, 'IncludeEncodedSetup.h')
    TEST_FEATURE_FILE = ""
    TEST_CLASSIFICATION_FILE = ""

    lines = [
        "#ifndef INCLUDEENCODEDSETUP_H",
        "#define INCLUDEENCODEDSETUP_H",
        f"#define THRESHOLD {threshold}",
        f"#define CLAUSES {clauses}",
        f"#define CLASSES {classes}",
        f"#define FEATURES {features}",
        f"#define NUMBER_OF_STATES {states}",
        "#define NUMBER_OF_TEST_EXAMPLES 0",
        "#define TEST_LENGTH_PER32 0",
        f'const char* TEST_FEATURE_FILE = "{TEST_FEATURE_FILE}";',
        f'const char* TEST_CLASSIFICATION_FILE = "{TEST_CLASSIFICATION_FILE}";',
        f"#define NUM_OF_INCLUDES {num_includes}",
        f'const char* INCENC_TA_FILE = "{encoded_filename}";',
        f'const char* INC_PER_CLASS_FILE = "{include_per_class_filename}";',
        "#define CLAUSE_POLARITY_BIT 15        // The bit holding information on clause polarity.",
        "#define BIT_DEMARCATING_CLAUSE_CHANGE 14    // The bit that demarcates beginning of next clause.",
        "#define INT_VALUE_OF_ENCODING_BITS 16383  // 14 bits used for literal offset",
        "#endif"
    ]

    with open(header_path, 'w') as fp:
        fp.write("\n".join(lines))

    print("Wrote:", header_path, flush=True)
    return header_path


def generate_data_store_h(dataset_name, automata, active_TAs_per_class, write_path='.'):
    """
    Generate a C++-friendly data_store.h containing ONLY:
      - static const uint16_t IncEncTA[]
      - static const uint16_t INC_per_CLASS[]
    No inference literals, no labels, no test-related macros.
    """

    outname = f"{dataset_name}_data_store.h"
    outpath = os.path.join(write_path, outname)

    # Convert arrays to comma-separated lists (C style)
    automata_list = ", ".join(str(int(x)) for x in automata)
    per_class_list = ", ".join(str(int(x)) for x in active_TAs_per_class)

    with open(outpath, 'w') as fp:
        fp.write("// Auto-generated compressed TA data for inference\n")
        fp.write("#pragma once\n\n")
        fp.write("#include <cstdint>\n\n")
        fp.write("namespace TMData {\n\n")

        fp.write(f"static const uint16_t IncEncTA[] = {{ {automata_list} }};\n\n")
        fp.write(f"static const uint16_t INC_per_CLASS[] = {{ {per_class_list} }};\n\n")

        fp.write(f"static constexpr std::size_t NUM_OF_INCLUDES = sizeof(IncEncTA) / sizeof(uint16_t);\n")
        fp.write(f"static constexpr std::size_t NUM_CLASSES = sizeof(INC_per_CLASS) / sizeof(uint16_t);\n\n")

        fp.write("} // namespace TMData\n")

    print("Wrote:", outpath, flush=True)
    return outpath



def main():
    parser = argparse.ArgumentParser(
        description="Encode TA states into REDRESS include-encoded TA (no test file required)."
    )

    parser.add_argument("dataset_name", help="Name for the dataset (used in output filenames)", type=str)
    parser.add_argument("threshold", help='Threshold value (T) used during TM training', type=int)
    parser.add_argument("clauses", help='Number of clauses used while training the TM', type=int)
    parser.add_argument("classes", help='Number of classes in the classification problem', type=int)
    parser.add_argument("features", help='Number of features in the problem (original features, not literals)', type=int)
    parser.add_argument("ta", help='TA states filename (space-separated numeric values)', type=str)
    parser.add_argument("-states", help="Number of states in TA (default: 200)", type=int, default=200)
    parser.add_argument("--write-setup", help="Write IncludeEncodedSetup.h with safe placeholders", action='store_true')
    parser.add_argument("--write-datastore", help="Write <dataset>_data_store.h containing IncEncTA and INC_per_CLASS (inference arrays empty)", action='store_true')
    parser.add_argument("--outdir", help="Directory to write output files (default: current dir)", type=str, default='.')

    args = parser.parse_args()

    # Validate TA file
    if not os.path.isfile(args.ta):
        print("ERROR: TA file not found:", args.ta,flush=True)
        sys.exit(1)

    # Load raw TA states (dtype unsigned short / H)
    raw_states = np.fromfile(args.ta, dtype=np.uint16, sep=' ')
    if raw_states is None:
        print("ERROR: Could not read TA file or file empty:", args.ta,flush=True)
        sys.exit(1)

    binary_states = parse_inc_exc_from_raw_states(raw_states, args.states)

    # number of literals: original script used features*2
    num_literals = int(args.features) * 2
    encoded_filename = os.path.join(args.outdir,
        f"{args.dataset_name}{args.features}_REDRESS_IncEncTA_C{args.classes}_CL{args.clauses}_T{args.threshold}.txt")
    include_per_class_filename = os.path.join(args.outdir,
        f"{args.dataset_name}{args.features}_REDRESS_Inc_per_class.txt")

    automata, active_TAs_per_class, num_includes = include_encoding(
        num_literals=int(num_literals),
        num_classes=int(args.classes),
        num_clauses=int(args.clauses),
        binary_states=binary_states,
        output_filename=encoded_filename,
        include_per_class_filename=include_per_class_filename
    )

    # Optionally generate setup header (with test placeholders)
    if args.write_setup:
        generate_include_encoded_setup_h(
            dataset_name=args.dataset_name,
            threshold=int(args.threshold),
            clauses=int(args.clauses),
            classes=int(args.classes),
            features=int(args.features),
            states=int(args.states),
            num_includes=int(num_includes),
            encoded_filename=os.path.basename(encoded_filename),
            include_per_class_filename=os.path.basename(include_per_class_filename),
            write_path=args.outdir
        )

    # Optionally generate microcontroller data store header (no test inputs included)
    if args.write_datastore:
        generate_data_store_h(args.dataset_name, automata, active_TAs_per_class, write_path=args.outdir)

    print("-----------------------------------------------------------------------", flush=True)
    print("DONE. Only TA compression completed (no test files required).", flush=True)
    print("-----------------------------------------------------------------------", flush=True)


if __name__ == "__main__":
    main()

#How to use
# python3 encode_ta_only_full.py MNIST 10 200 10 784 ./TA_files/MNIST_TAs.txt
# python3 encode_ta_only_full.py MNIST 10 200 10 784 ./TA_files/MNIST_TAs.txt --write-setup
# python3 encode_ta_only_full.py MNIST 10 200 10 784 ./TA_files/MNIST_TAs.txt --write-datastore
# python3 encode_ta_only_full.py MNIST 10 200 10 784 ./TA_files/MNIST_TAs.txt --write-setup --outdir out
