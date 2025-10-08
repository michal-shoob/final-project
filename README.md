# Learning GRNs


---

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/michal-shoob/final-project.git
   cd final-project
    ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
project_name/
│
├── src/                      # Source code
│   ├── main.py               # Main entry point
│   ├── boolean_network.py    # Boolean network implementation
│   ├── td_agent.py           # TD agent implementation
│   ├── logger_setup.py       # Logging setup
│   └── utils/                # Utility functions or modules 
│
├── tests/                    # Unit and integration tests
│
├── data/                     # Datasets or input files (if any)
│
├── results/                  # Output files and visualizations
│   ├── network_states/       # State evolution diagrams
│   └── value_tables/         # Learned value-table visualizations
|--- Screenshots              # Output 1 : network_states, table_visualization
│
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation (this file)
```

---

## Usage

Run the project using the following steps:

1. **Basic Command**:
   ```bash
   python src/main.py
   ```

2. **Optional Arguments** (if applicable):
   ```bash
   python src/main.py --arg1 value1 --arg2 value2
   ```

   3. **Example**:
    ```bash
   python src/main.py --num_episodes 1000 --learning_rate 0.1
   ```
   This command runs the TD agent for 1000 episodes with a learning rate of 0.1.

---