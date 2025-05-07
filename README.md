# 2048 Q-Learning Agent

Questo progetto esplora vari approcci per sviluppare un agente per il gioco **2048** utilizzando il **Q-Learning**.

## 🎯 Obiettivo

Implementare e addestrare un agente in grado di giocare a 2048 apprendendo tramite Q-Learning, confrontando diverse strategie, iperparametri e architetture di stato.

## Video Demo

[![Demo 2048](https://img.youtube.com/vi/-vxgao3myio/0.jpg)](https://youtu.be/-vxgao3myio)

## 🚀 Caratteristiche

* Ambiente di gioco 2048 implementato in Python
* Agente basato su Q-Learning con funzioni di utilità personalizzabili
* Supporto per tuning di iperparametri (learning rate, discount factor, epsilon-greedy)
* Script per addestramento e valutazione automatica
* Logging e visualizzazione delle performance durante l'allenamento

## 📦 Requisiti

* Python 3.10
* Vedere `requirements.txt` per le librerie Python necessarie

## 🛠️ Installazione

```bash
# Clona il repository
git clone <https://github.com/Rocco9999/2048_Q-Learning>
cd 2048-qlearning-agent

# Crea e attiva l'ambiente virtuale (Windows)
python -m venv venv
.\venv\Scripts\activate

# (Unix/MacOS)
# python3 -m venv venv
# source venv/bin/activate

# Installa le dipendenze
pip install -r requirements.txt
```

## 📋 Struttura del Progetto

```
2048-qlearning-agent/
├── agent.py              # Implementazione dell'agente Q-Learning
├── environment.py        # Definizione dell'ambiente di gioco 2048
├── train.py              # Script per l'addestramento dell'agente
├── evaluate.py           # Script per la valutazione e il testing
├── requirements.txt      # Dipendenze Python
├── README.md             # Documentazione del progetto
└── models/               # Directory per salvare i modelli addestrati
```

## ⚙️ Uso

### Addestramento

```bash
python train.py \
  --episodes 10000 \
  --alpha 0.1 \
  --gamma 0.99 \
  --epsilon 0.95
```

Parametri principali:

* `--episodes`: numero di epoche di allenamento
* `--alpha`: learning rate
* `--gamma`: fattore di sconto
* `--epsilon`: probabilità di esplorazione iniziale (epsilon-greedy)

## 📈 Visualizzazione dei Risultati

Durante l'allenamento, metriche come punteggio medio e massimale vengono registrate in `logs/`. 

