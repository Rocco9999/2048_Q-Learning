import pandas as pd
import glob
import ast
import os

# Funzione per convertire Q-Values da stringa a lista di float
def convert_q_values(value):
    try:
        return [float(x) for x in ast.literal_eval(value)]
    except (ValueError, SyntaxError):
        return None  # Gestisce valori non validi

# Carica tutti i file CSV nella directory
files = glob.glob("C:/Users/rocco/OneDrive/Desktop/Università/IA/2048_Q-Learning/*.csv")

dataframes = []
for file in files:
    df = pd.read_csv(file)
    
    # Converte la colonna Q-Values
    if 'Q-Values' in df.columns:
        df['Q-Values'] = df['Q-Values'].apply(convert_q_values)
    
    # Aggiunge il nome del file come identificatore della tecnica
    df['Reward_Technique'] = file.split("\\")[-1].split(".csv")[0]
    dataframes.append(df)

# Combina tutti i file in un unico DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Analisi
summary_data = []

# Calcola statistiche per reward
for technique, group in combined_df.groupby('Reward_Technique'):
    avg_reward = group['Reward'].mean()
    std_reward = group['Reward'].std()
    max_value = group['Max Value'].max()
    
    # Calcola distribuzione delle azioni
    action_distribution = group['Action'].value_counts(normalize=False).to_dict()
    
    # Salva i dati nel riepilogo
    summary_row = {
        'Reward_Technique': technique,
        'Avg_Reward': avg_reward,
        'Std_Reward': std_reward,
        'Max_Value': max_value
    }
    
    # Aggiungi distribuzione delle azioni al riepilogo
    for action, count in action_distribution.items():
        summary_row[f'Action_{int(action)}'] = count
    
    summary_data.append(summary_row)

# Crea un DataFrame per il riepilogo
summary_df = pd.DataFrame(summary_data)

# Ordina le colonne per avere le azioni in ordine
action_columns = [col for col in summary_df.columns if col.startswith('Action_')]
summary_df = summary_df[['Reward_Technique', 'Avg_Reward', 'Std_Reward', 'Max_Value'] + sorted(action_columns)]

# Salva il riepilogo in CSV
save_dir = "C:/Users/rocco/OneDrive/Desktop/Università/IA/2048_Q-Learning/plots/"
os.makedirs(save_dir, exist_ok=True)
summary_csv_path = os.path.join(save_dir, "summary_statistics_cleaned.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Dati di riepilogo salvati in: {summary_csv_path}")
