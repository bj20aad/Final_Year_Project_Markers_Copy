import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
INPUT_FILE   = 'data/DNN-EdgeIIoT-dataset.csv'
TRAIN_OUTPUT = 'data/clean_benign_train.csv'
TEST_OUTPUT  = 'data/clean_mixed_test.csv'
CHUNK_SIZE   = 50000

# Text, identifier, and timestamp columns excluded from feature set
DROP_COLS = [
    'frame.time', 'ip.src_host', 'ip.dst_host',
    'arp.src.proto_ipv4', 'arp.dst.proto_ipv4',
    'http.file_data', 'http.request.full_uri', 'http.request.uri.query',
    'http.request.method', 'http.referer', 'http.request.version',
    'icmp.transmit_timestamp',
    'tcp.options', 'tcp.payload', 'tcp.srcport', 'tcp.dstport',
    'udp.port',
    'mqtt.msg', 'mqtt.protoname', 'mqtt.topic', 'mqtt.conack.flags',
    'dns.qry.name.len'
]


def preprocess_stream():
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: {INPUT_FILE} not found. Ensure the file is in the data/ folder.")
        return

    # Remove any existing output files before writing
    for path in [TRAIN_OUTPUT, TEST_OUTPUT]:
        if os.path.exists(path):
            os.remove(path)

    print(f"Processing {INPUT_FILE} in chunks of {CHUNK_SIZE}...")

    total_train_rows = 0
    reader = pd.read_csv(INPUT_FILE, chunksize=CHUNK_SIZE, low_memory=False)

    for i, chunk in enumerate(reader):
        # Drop excluded columns where present in DROP_COLS array above
        chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], inplace=True)

        # Retain only numeric features plus the label column
        if 'Attack_type' in chunk.columns:
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns.tolist()
            chunk = chunk[numeric_cols + ['Attack_type']]

        chunk.fillna(0, inplace=True)

        # Split: benign-only for unsupervised training, 20% random sample for evaluation
        if 'Attack_type' in chunk.columns:
            benign_chunk = chunk[chunk['Attack_type'] == 'Normal'].copy()
            test_chunk   = chunk.sample(frac=0.2, random_state=42)
        else:
            benign_chunk = chunk
            test_chunk   = chunk

        # Remove label from training set — unsupervised models train on features only
        benign_chunk.drop(columns=['Attack_type', 'Attack_label'], errors='ignore', inplace=True)

        # Append to output files (header written on first chunk only)
        header = (i == 0)

        if not benign_chunk.empty:
            benign_chunk.to_csv(TRAIN_OUTPUT, mode='a', header=header, index=False)
            total_train_rows += len(benign_chunk)

        if not test_chunk.empty:
            test_chunk.to_csv(TEST_OUTPUT, mode='a', header=header, index=False)

        print(f"  Chunk {i + 1} processed — training rows so far: {total_train_rows:,}")

    print("\nPreprocessing complete.")
    print(f"  Training set : {TRAIN_OUTPUT}  ({total_train_rows:,} rows)")
    print(f"  Evaluation set: {TEST_OUTPUT}")


if __name__ == "__main__":
    preprocess_stream()
