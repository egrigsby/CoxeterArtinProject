import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_FILE = Path("NFdata (1).csv")
OUTPUT_FILE = Path("processed_descent_data.csv")
SEQUENCE_LENGTH = 15  # Adjust based on your maximum word length

# Character-to-Integer Mapping (0 is strictly reserved for padding)
CHAR_MAP = {'a': 1, 'b': 2, 'c': 3}

def calculate_bitmask(descents_string):
    """Converts a string of characters like 'bc' into a single bitmask integer."""
    if pd.isna(descents_string):
        return 0
    
    mask_value = 0
    descents_string = str(descents_string).strip()
    for char in descents_string:
        if char in CHAR_MAP:
            gen_id = CHAR_MAP[char]
            # Bit 0 for 'a' (id 1), Bit 1 for 'b' (id 2), Bit 2 for 'c' (id 3)
            mask_value |= (1 << (gen_id - 1))
    return mask_value

def convert_row(word, final_descents, max_len):
    if pd.isna(word):
        word = ""
    word = str(word).strip()
    
    # 1. Convert characters to token IDs
    word_tokens = [CHAR_MAP.get(char, 0) for char in word]
    actual_len = len(word_tokens)
    
    # 2. Build descents: 0 for all early prefixes, and the actual bitmask for the final letter
    # (If the word is empty, we just make it an empty list)
    if actual_len > 0:
        descent_bitmasks = [0] * (actual_len - 1) + [calculate_bitmask(final_descents)]
    else:
        descent_bitmasks = []
        
    # 3. Apply Padding to match SEQUENCE_LENGTH
    if actual_len < max_len:
        padding_count = max_len - actual_len
        word_tokens += [0] * padding_count          # Pad words with 0
        descent_bitmasks += [-1] * padding_count    # Pad descents with -1
    else:
        word_tokens = word_tokens[:max_len]
        descent_bitmasks = descent_bitmasks[:max_len]
        
    return word_tokens, descent_bitmasks

if __name__ == "__main__":
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please put your raw data file in the same folder.")
        exit(1)
    
    # Clean up column names (removes any accidental spaces)
    df.columns = df.columns.str.strip()
    
    words_out = []
    descents_out = []
    
    print("Formatting rows...")
    for _, row in df.iterrows():
        tokens, bitmasks = convert_row(
            row['Word (in Normal Form)'], 
            row['Descents'], 
            SEQUENCE_LENGTH
        )
        words_out.append(str(tokens))
        descents_out.append(str(bitmasks))
        
    # Save to the exact schema Transformer-7.py looks for
    output_df = pd.DataFrame({
        "word": words_out,
        "descents": descents_out
    })
    
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Done! Saved perfectly formatted data to: {OUTPUT_FILE}")
