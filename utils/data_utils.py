from typing import List, Dict, Tuple
import pandas as pd

def load_data() -> pd.DataFrame:
    import pandas as pd

    base_url = "https://raw.githubusercontent.com/BobAdamsEE/SouthParkData/refs/heads/master/by-season/Season-{}.csv"
    dfs = []
    for season in range(1, 20):  # saisons 1 to 19
        url = base_url.format(season)
        dfs.append(pd.read_csv(url))
        
    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_dial = df.iloc[:,-2:]
    df_dial["Line"] = df_dial["Line"].str.replace("\n", " ", regex=False).str.strip()
    print(f"Original data: {len(df_dial):,} lines")
    # Remove duplicates
    df_clean = df_dial.drop_duplicates()
    print(f"After removing duplicates: {len(df_clean):,} lines")


    # Strip whitespace from character names
    df_clean['Character'] = df_clean['Character'].str.strip()
    df_clean['Line'] = df_clean['Line'].str.strip()

    # Remove empty lines
    df_clean = df_clean[df_clean['Line'].str.len() > 0]
    print(f"After removing empty lines: {len(df_clean):,} lines")
    
    # Select top K characters by number of lines
    character_counts = df_clean['Character'].value_counts()
    top_characters = character_counts[character_counts > 200].index.tolist()  # Keep characters with more than 200 lines

    print(f"\nTop {len(top_characters)} characters:")
    for i, char in enumerate(top_characters, 1):
        count = character_counts[char]
        print(f"  {i:2d}. {char:20s} - {count:5d} lines ({count/len(df_clean)*100:.1f}%)")

    # Filter dataset to only include top K characters
    df_model = df_clean[df_clean['Character'].isin(top_characters)].copy()
    print(f"\nFiltered dataset: {len(df_model):,} lines ({len(df_model)/len(df_clean)*100:.1f}% of total)")

    return df_model

def create_character_mapping(df, top_characters: List[str]):
    """Create character to label mapping."""
    # Create character to label mappings
    char_to_label = {char: idx for idx, char in enumerate(top_characters)}
    label_to_char = {idx: char for char, idx in char_to_label.items()}

    print("Character to label mapping:")
    for char, label in char_to_label.items():
        print(f"  {label}: {char}")

    return char_to_label, label_to_char