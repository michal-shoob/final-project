# Learning GRNs

Temporal-difference (TD) learning agent that searches for desirable states in
Boolean gene regulatory networks (GRNs). The project wires domain-specific
network definitions (`boolean_network.py`) into a configurable TD agent
(`td_agent.py`) and exposes a minimal CLI via `main.py`.

---

## Prerequisites

- Python 3.10+
- Git (if you plan to clone the repository)
- A C compiler is recommended because `pyboolnet` installs from source

---

## Setup

1. **Clone or copy the project**
   ```bash
   git clone https://github.com/michal-shoob/final-project.git
   cd final-project
   ```
2. **(Optional) Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # PowerShell on Windows
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This pulls `pyboolnet` directly from GitHub plus the Python packages used for
   learning, plotting, and analysis.

---

## Project Structure

```
.
├── agent.py                # Early agent experiments (unused in main script)
├── boolean_network.py      # GRN definitions, edge functions, targets
├── helper.py               # Utility helpers (e.g., target computation)
├── logger_setup.py         # Logging configuration shared across modules
├── main.py                 # Entry point wrapping the TD agent
├── td_agent.py             # Implementation of the TD learning agent
├── requirements.txt        # Python dependencies
├── network_states.png      # Example visualization of explored states
├── table_visualization.png # Example learned value table
└── README.md
```

---

## Usage

Run the main entry point after installing dependencies:

```bash
python main.py --epsilon 0.1
```

**Arguments**

- `--epsilon` (float, default `0.1`): Exploration rate passed to `TDAgent`.

The script currently trains on the 50-node GRN defined in
`boolean_network.py`. You can switch to the 100-node configuration by
uncommenting the corresponding lines in `main.py`.

**Outputs**

- Logs stream to the console and `app.log` (via `logger_setup.py`).
- Visual artifacts such as `network_states.png` and `table_visualization.png`
  summarize recent experiments.

---

## Project Documentation

[View the Capstone Project Report here](<https://docs.google.com/document/d/1gG3AKNKqOxGFPkRrkZjE7LSmMy7QIRskd2ZmLe1ufIY/edit?usp=sharing>)

[View the Project Presentation Slides here](<https://drive.google.com/file/d/1PqpRwxh1FKo4S4KCv4T2siwJMum6GMsD/view?usp=sharing>)

---

