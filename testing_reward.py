import pandas as pd
import glob
import ast
import matplotlib.pyplot as plt
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
    df['Reward_Technique'] = file.split("/")[-1].split(".csv")[0]
    dataframes.append(df)

# Combina tutti i file in un unico DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Salvataggio directory
save_dir = "C:/Users/rocco/OneDrive/Desktop/Università/IA/2048_Q-Learning/plots/"

os.makedirs(save_dir, exist_ok=True)

# Visualizzazione 1: Trend delle ricompense medie
for technique, group in combined_df.groupby('Reward_Technique'):
    group.groupby('Episode')['Reward'].mean().plot(label=technique)

plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Reward Trend per Episode")
plt.legend()
plt.savefig(f"{save_dir}reward_trend.png")  # Salva il grafico
plt.show()

# Visualizzazione 2: Distribuzione del valore massimo per tecnica
combined_df.boxplot(column='Max Value', by='Reward_Technique')
plt.title("Max Value Distribution by Reward Technique")
plt.ylabel("Max Value")
plt.xlabel("Reward Technique")
plt.suptitle("")  # Rimuove il titolo predefinito di boxplot
plt.savefig(f"{save_dir}max_value_distribution.png")  # Salva il grafico
plt.show()

# Visualizzazione 3: Distribuzione delle azioni
action_distribution = combined_df.groupby(['Reward_Technique', 'Action']).size()
action_distribution.unstack().plot(kind='bar', stacked=True)
plt.title("Action Distribution by Reward Technique")
plt.xlabel("Reward Technique")
plt.ylabel("Action Count")
plt.savefig(f"{save_dir}action_distribution.png")  # Salva il grafico
plt.show()

# Visualizzazione 4: Massimo valore raggiunto per ogni tecnica
max_values = combined_df.groupby('Reward_Technique')['Max Value'].max()
max_values.plot(kind='bar', color='skyblue')
plt.title("Max Value Reached by Reward Technique")
plt.xlabel("Reward Technique")
plt.ylabel("Max Value")
plt.savefig(f"{save_dir}max_value_reached.png")  # Salva il grafico
plt.show()

# Visualizzazione 5: Andamento del Max Value per Episodio (uno per tecnica)
techniques = combined_df['Reward_Technique'].unique()
n_techniques = len(techniques)

# Creazione di grafici separati per ogni tecnica
for technique, group in combined_df.groupby('Reward_Technique'):
    group = group.sort_values('Episode')
    max_values_per_episode = group.groupby('Episode')['Max Value'].max()
    
    plt.figure(figsize=(10, 6))
    plt.plot(max_values_per_episode, label=technique)
    plt.title(f"Reward Technique: {technique}")
    plt.xlabel("Episode")
    plt.ylabel("Max Value")
    plt.legend()
    otherName= technique.split("\\")[1]
    plt.savefig(f"{save_dir}{otherName}_max_value_trend.png")  # Salva il grafico
    plt.close()  # Chiude il grafico per evitare sovrapposizioni
