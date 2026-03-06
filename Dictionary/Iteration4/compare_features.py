import pandas as pd
from data_loader import load_and_preprocess_data

def print_top_values(df, col_name, top_n=5):
    print(f"\n--- Top {top_n} {col_name} ---")
    
    print("BENIGN (0):")
    b_counts = df[df['label'] == 0][col_name].value_counts()
    for val, count in list(b_counts.items())[:top_n]:
        print(f"  {val}: {count} ({count/len(df[df['label']==0])*100:.1f}%)")
        
    print("BOTS (1):")
    bot_counts = df[df['label'] == 1][col_name].value_counts()
    for val, count in list(bot_counts.items())[:top_n]:
        print(f"  {val}: {count} ({count/len(df[df['label']==1])*100:.1f}%)")
        
    print("MALWARE (2):")
    m_counts = df[df['label'] == 2][col_name].value_counts()
    for val, count in list(m_counts.items())[:top_n]:
        print(f"  {val}: {count} ({count/len(df[df['label']==2])*100:.1f}%)")

if __name__ == "__main__":
    df = load_and_preprocess_data("../Dictionary/ja4+_db.json")
    
    features = [
        "ja4_a_protocol", "ja4_a_tls", "ja4_a_sni", 
        "ja4_a_cipher_cnt", "ja4_a_ext_cnt", "ja4_a_alpn",
        "ja4_b", "ja4_c"
    ]
    
    for f in features:
        print_top_values(df, f)
