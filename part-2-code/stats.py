import os
import numpy as np
from transformers import T5TokenizerFast
from tqdm import tqdm
import warnings
from load_data import load_t5_data
# Suppress tokenizer warnings
warnings.filterwarnings("ignore", "Token indices sequence length is longer than.*")

# --- Configuration ---
DATA_FOLDER = "data"
TOKENIZER_NAME = "google-t5/t5-small"

# --- Helper Functions ---

def load_raw_lines(split):
    """Loads raw text lines from .nl and .sql files."""
    nl_path = os.path.join(DATA_FOLDER, f"{split}.nl")
    sql_path = os.path.join(DATA_FOLDER, f"{split}.sql")
    
    if not os.path.exists(nl_path) or not os.path.exists(sql_path):
        return None, None
        
    with open(nl_path, "r") as f:
        nl_lines = [l.strip() for l in f.readlines()]
    with open(sql_path, "r") as f:
        sql_lines = [l.strip() for l in f.readlines()]
    return nl_lines, sql_lines

def get_stats_from_ids(token_id_lists):
    """Calculates mean length and vocab size from lists of token IDs."""
    if not token_id_lists:
        return 0.0, 0
        
    lengths = [len(ids) for ids in token_id_lists]
    mean_len = np.mean(lengths)
    
    vocab = set()
    for ids in token_id_lists:
        vocab.update(ids)
    vocab_size = len(vocab)
    
    return mean_len, vocab_size

# --- Main Calculation ---

def main():
    try:
        tokenizer = T5TokenizerFast.from_pretrained(TOKENIZER_NAME)
    except Exception as e:
        print(f"Error loading tokenizer '{TOKENIZER_NAME}'.")
        print("Please ensure you have internet access and the huggingface_hub library.")
        print(f"Error: {e}")
        return

    print("--- Calculating Statistics for Table 1 (Before Preprocessing) ---")
    
    train_nl_raw, train_sql_raw = load_raw_lines("train")
    dev_nl_raw, dev_sql_raw = load_raw_lines("dev")
    
    if train_nl_raw is None:
        print(f"Error: Could not find 'train' data in '{DATA_FOLDER}' directory.")
        return
    if dev_nl_raw is None:
        print(f"Error: Could not find 'dev' data in '{DATA_FOLDER}' directory.")
        return

    # Tokenize raw text without truncation or special tokens for Table 1
    train_nl_ids_raw = tokenizer(train_nl_raw, add_special_tokens=False).input_ids
    train_sql_ids_raw = tokenizer(train_sql_raw, add_special_tokens=False).input_ids
    dev_nl_ids_raw = tokenizer(dev_nl_raw, add_special_tokens=False).input_ids
    dev_sql_ids_raw = tokenizer(dev_sql_raw, add_special_tokens=False).input_ids
    
    # Get stats
    tr_nl_mean, tr_nl_vocab = get_stats_from_ids(train_nl_ids_raw)
    tr_sql_mean, tr_sql_vocab = get_stats_from_ids(train_sql_ids_raw)
    dev_nl_mean, dev_nl_vocab = get_stats_from_ids(dev_nl_ids_raw)
    dev_sql_mean, dev_sql_vocab = get_stats_from_ids(dev_sql_ids_raw)

    print("\n[COPY-PASTE FOR TABLE 1]")
    print(f"| Statistics Name                | Train         | Dev           |")
    print(f"|--------------------------------|---------------|---------------|")
    print(f"| Number of examples             | {len(train_nl_raw):<13} | {len(dev_nl_raw):<13} |")
    print(f"| Mean sentence length           | {tr_nl_mean:<13.2f} | {dev_nl_mean:<13.2f} |")
    print(f"| Mean SQL query length          | {tr_sql_mean:<13.2f} | {dev_sql_mean:<13.2f} |")
    print(f"| Vocabulary size (natural language) | {tr_nl_vocab:<13} | {dev_nl_vocab:<13} |")
    print(f"| Vocabulary size (SQL)          | {tr_sql_vocab:<13} | {dev_sql_vocab:<13} |")
    
    
    print("\n--- Calculating Statistics for Table 2 (After Preprocessing) ---")
    
    # Replicate the *exact* preprocessing from your T5Dataset
    MAX_SOURCE_LEN = 256
    MAX_TARGET_LEN = 256
    BOS_ID = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    EOS_ID = tokenizer.eos_token_id

    def process_split_for_table2(nl_lines, sql_lines):
        processed_enc_ids = []
        processed_target_ids = []
        
        for nl, sql in zip(nl_lines, sql_lines):
            # 1. Process Encoder (NL)
            enc = tokenizer(
                nl,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SOURCE_LEN,
            )
            processed_enc_ids.append(enc["input_ids"])
            
            # 2. Process Decoder (SQL)
            sql_enc = tokenizer(
                sql,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_TARGET_LEN - 2, # As per your script
            )
            
            # Create the final target sequence: sql_ids + [EOS]
            processed_target_ids.append(sql_enc["input_ids"] + [EOS_ID])
            
        return processed_enc_ids, processed_target_ids

    tr_enc_ids_pp, tr_target_ids_pp = process_split_for_table2(train_nl_raw, train_sql_raw)
    dev_enc_ids_pp, dev_target_ids_pp = process_split_for_table2(dev_nl_raw, dev_sql_raw)

    # Get stats
    tr_nl_mean_pp, tr_nl_vocab_pp = get_stats_from_ids(tr_enc_ids_pp)
    tr_sql_mean_pp, tr_sql_vocab_pp = get_stats_from_ids(tr_target_ids_pp)
    dev_nl_mean_pp, dev_nl_vocab_pp = get_stats_from_ids(dev_enc_ids_pp)
    dev_sql_mean_pp, dev_sql_vocab_pp = get_stats_from_ids(dev_target_ids_pp)

    print("\n[COPY-PASTE FOR TABLE 2]")
    print(f"| Statistics Name                | Train         | Dev           |")
    print(f"|--------------------------------|---------------|---------------|")
    print(f"| Model name                     | google-t5/t5-small             |")
    print(f"| Mean sentence length           | {tr_nl_mean_pp:<13.2f} | {dev_nl_mean_pp:<13.2f} |")
    print(f"| Mean SQL query length          | {tr_sql_mean_pp:<13.2f} | {dev_sql_mean_pp:<13.2f} |")
    print(f"| Vocabulary size (natural language) | {tr_nl_vocab_pp:<13} | {dev_nl_vocab_pp:<13} |")
    print(f"| Vocabulary size (SQL)          | {tr_sql_vocab_pp:<13} | {dev_sql_vocab_pp:<13} |")
    print("\n(Note: 'Mean sentence length' in Table 2 refers to the final truncated encoder input.)")
    print("(Note: 'Mean SQL query length' in Table 2 refers to the final truncated *target* sequence.)")

if __name__ == "__main__":
    main()