# MLflow Experiment Tracking with ElasticNet Regression

A hands-on Machine Learning Operations (MLOps) project demonstrating how to train a machine learning model, evaluate its performance, track experiments, log parameters and metrics, and perform hyperparameter tuning using **MLflow** and **scikit-learn**.

This project uses the UCI Wine dataset and implements ElasticNet regression to explore how different hyperparameter combinations affect model performance.

## 📌 Project Overview

Experiment tracking is an essential part of the machine learning lifecycle. It helps data scientists and ML engineers record training configurations, evaluate model performance, compare different runs, and maintain a reproducible record of experiments.

In this project, MLflow is used to track experiments for an ElasticNet model trained with different values of `alpha` and `l1_ratio`.

### Key Features

- **Model training:** Train an ElasticNet model using scikit-learn.
- **Dataset retrieval:** Fetch the Wine dataset using `ucimlrepo`.
- **Data splitting:** Split the dataset into training and testing sets.
- **Model evaluation:** Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).
- **Experiment tracking:** Track model parameters, evaluation metrics, and the trained model using MLflow.
- **Hyperparameter tuning:** Run multiple experiments with different `alpha` and `l1_ratio` values.
- **Experiment comparison:** Compare results through the MLflow web-based user interface.
- **Model logging:** Save the trained model as an MLflow model artifact.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| MLflow | Experiment tracking and model logging |
| scikit-learn | Model training and evaluation |
| NumPy | Generate hyperparameter combinations |
| Pandas | Data handling |
| ucimlrepo | Retrieve the UCI Wine dataset |
| Conda | Python environment management |

## 📂 Project Structure

```text
mlflow-elasticnet-experiment-tracking/
│
├── sim_model.py       # Data retrieval, model training and MLflow logging
├── run.py             # Multiple experiments and hyperparameter tuning
├── README.md          # Project documentation
├── .gitignore         # Excludes generated and local files
│
└── mlruns/            # MLflow experiment and run artifacts (generated locally)
```

**Note:** The `mlruns/` directory and the local MLflow database are generated during execution. They should generally be excluded from Git to keep the repository clean.

## ⚙️ Prerequisites

Before running this project, make sure you have installed:

- Python 3.12
- Anaconda or Miniconda
- Git
- A code editor such as Visual Studio Code

An internet connection is required to download the dataset and install Python packages.

## 🚀 Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mlflow-elasticnet-experiment-tracking.git
cd mlflow-elasticnet-experiment-tracking
```

Replace `YOUR_USERNAME` with your GitHub username.

Alternatively, download the repository as a ZIP file and extract it.

### Step 2: Create a Conda environment

Open Command Prompt or the VS Code terminal and execute:

```bash
conda create --name myenv python=3.12.0
```

Activate the environment:

```bash
conda activate myenv
```

Verify the Python version:

```bash
python --version
```

Expected version:

```text
Python 3.12.0
```

### Step 3: Install the dependencies

Install the required libraries:

```bash
pip install mlflow
pip install ucimlrepo
pip install pandas numpy scikit-learn
```

Verify the MLflow installation:

```bash
mlflow --version
```

## 🧠 Understanding the Project

### 1. Data retrieval

The `get_data()` function retrieves the Wine dataset from the UCI Machine Learning Repository using:

```python
wine = fetch_ucirepo(id=109)

x = wine.data.features
y = wine.data.targets
```

- `x`: Input features used for training.
- `y`: Target values the model attempts to predict.

The dataset is fetched when the training function runs.

### 2. Train-test split

The dataset is divided into training and testing subsets:

```python
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=21
)
```

- 80% of the data is used for training.
- 20% is used for testing.
- `random_state=21` makes the split reproducible for the same dataset and environment.

### 3. ElasticNet model

ElasticNet is a linear model that combines L1 and L2 regularization.

```python
en = ElasticNet(
    alpha=alpha,
    l1_ratio=l1_ratio,
    random_state=21
)
```

The main hyperparameters are:

| Parameter | Description |
|---|---|
| `alpha` | Controls the overall regularization strength. Larger values generally apply stronger regularization. |
| `l1_ratio` | Controls the balance between L1 and L2 regularization. A value of 1 corresponds to Lasso, while 0 corresponds to Ridge-style L2 regularization. |
| `random_state` | Controls randomness where applicable, supporting reproducibility. |

The model learns from the training data:

```python
en.fit(x_train, y_train)
```

It then generates predictions for the test data:

```python
pred = en.predict(x_test)
```

### 4. Model evaluation

The `evaluate_model()` function calculates three evaluation metrics.

**Mean Squared Error (MSE)**

Measures the average squared difference between actual and predicted values. Lower values indicate smaller squared prediction errors.

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values. Lower values indicate smaller average errors.

**R-squared (R²)**

Measures how well the model explains the variation in the target values. A value closer to 1 generally indicates a better fit; negative values are possible.

The metrics are calculated using:

```python
mse = mean_squared_error(actual, predicted)
mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)
```

### 5. MLflow experiment tracking

The project creates or selects an MLflow experiment named `simple_model`:

```python
mlflow.set_experiment("simple_model")
```

The model training process runs inside an MLflow run:

```python
with mlflow.start_run(run_name="ElasticNet_Final"):
    ...
```

Within each run, the following information is logged.

**Parameters**

```python
params = {
    "alpha": alpha,
    "l1_ratio": l1_ratio
}

mlflow.log_params(params)
```

**Metrics**

```python
metrics = {
    "mse": mse,
    "mae": mae,
    "r2": r2
}

mlflow.log_metrics(metrics)
```

**Model artifact**

```python
mlflow.sklearn.log_model(
    sk_model=en,
    name="model"
)
```

This records the trained scikit-learn model as an MLflow model artifact.

Each run stores its own parameters, metrics, and model artifact, allowing you to compare the results of different configurations.

## ▶️ Running a Single Experiment

The `sim_model.py` file trains one ElasticNet model with a fixed hyperparameter configuration.

Default values:

```python
alpha = 0.05
l1_ratio = 0.05
```

Run the script:

```bash
python sim_model.py
```

The terminal prints the selected hyperparameters and the evaluation metrics.

Example output format:

```text
ElasticNet model parameters (alpha=0.05, l1_ratio=0.05):
ElasticNet model performance:
  MSE = <calculated value>
  MAE = <calculated value>
  R2 = <calculated value>
```

The numerical values depend on the dataset and installed library versions; they are intentionally not hardcoded here.

## 📊 Viewing Results in the MLflow UI

### Step 1: Start the MLflow UI

Open a second terminal in the same project directory and activate the same Conda environment:

```bash
conda activate myenv
```

Start the UI:

```bash
mlflow ui
```

By default, MLflow starts its local tracking interface on port 5000.

### Step 2: Open the browser

Navigate to either address:

- http://127.0.0.1:5000
- http://localhost:5000

### Step 3: Explore your experiment

In the MLflow interface:

1. Open the `simple_model` experiment.
2. Select a run to inspect its parameters and metrics.
3. Compare the values of `alpha` and `l1_ratio`.
4. Review MSE, MAE, and R².
5. Open the logged model artifact to inspect its MLflow model files.

Keep the terminal running while you use the interface.

If the page is blank at `localhost:5000`, try `http://127.0.0.1:5000` and refresh the page.

## 🔁 Running Multiple Experiments

The `run.py` script executes multiple training runs using different hyperparameter combinations.

The hyperparameter values are generated using NumPy:

```python
alpha_values = np.linspace(0.1, 0.5, 5)
l1_ratio_values = np.linspace(0.1, 0.5, 5)
```

Both arrays contain five values:

```text
[0.1, 0.2, 0.3, 0.4, 0.5]
```

Nested loops evaluate every combination:

```python
for alpha in alpha_values:
    for l1_ratio in l1_ratio_values:
        train(alpha, l1_ratio)
```

This produces:

**5 alpha values × 5 l1_ratio values = 25 training runs**

Each call to `train()` starts a separate MLflow run within the `simple_model` experiment.

### Execute the experiments

With the MLflow UI running in another terminal, execute:

```bash
python run.py
```

After the script finishes, refresh the MLflow UI.

You should see 25 additional runs from this hyperparameter sweep, assuming the runs complete successfully.

### Comparing model performance

In the MLflow UI, compare the runs using:

- `alpha`
- `l1_ratio`
- `mse`
- `mae`
- `r2`

For this regression evaluation, lower MSE and MAE are generally preferable. Higher R² is generally preferable when the metric is meaningful for the target.

You can sort or compare runs using these logged values to identify promising hyperparameter configurations.

**Important:** The `run.py` script does not automatically select or register the best model. It generates runs for manual comparison in the MLflow UI.

## 📈 Experiment Workflow

```text
Start
  |
  v
Fetch Wine Dataset
  |
  v
Split into Training and Testing Data
  |
  v
Initialize ElasticNet
  |
  v
Train the Model
  |
  v
Generate Predictions
  |
  v
Calculate MSE, MAE and R²
  |
  v
Log Parameters, Metrics and Model to MLflow
  |
  v
Compare Runs in MLflow UI
```

For multiple experiments, the hyperparameter loop repeats this workflow for each combination.

## 🧹 Git Ignore Configuration

The `.gitignore` file can contain:

```gitignore
# IDE
.vscode/

# Python
__pycache__/
*.py[cod]

# Virtual environments
.venv/
venv/

# MLflow generated data
mlruns/
mlflow.db
simple_model/mlflow.db

# Environment variables
.env
```

These exclusions prevent local configuration, generated MLflow tracking data, and environment files from being committed unintentionally.

## ⚠️ Important Notes and Improvements

- **Target type:** The UCI Wine dataset is commonly used for classification. This project uses `ElasticNet`, which is a regression estimator. If the target contains wine class labels, the resulting regression metrics do not represent standard classification performance. For a classification task, consider a classifier and metrics such as accuracy, precision, recall, and F1-score.
- **Feature scaling:** ElasticNet regularization is sensitive to feature scales. Consider using a scikit-learn `Pipeline` with `StandardScaler` for a more reliable modeling workflow.
- **Run names:** Every run currently uses `run_name="ElasticNet_Final"`. Consider generating descriptive names containing the hyperparameters to make runs easier to identify.
- **Data retrieval:** `get_data()` downloads or retrieves the dataset each time `train()` is called. For a more efficient sweep, load the dataset once and reuse it across all 25 runs.
- **Reproducibility:** The fixed random state helps keep the data split consistent, but reproducibility can also depend on package versions and dataset availability.
- **Tracking storage:** The default `mlflow ui` command uses local tracking data. Keep the working directory consistent so the UI reads the same tracking store used by the training scripts.

## 🎯 Learning Outcomes

After completing this project, you will understand how to:

- Set up a Python environment for MLflow.
- Train and evaluate a scikit-learn model.
- Create and manage MLflow experiments.
- Log model parameters and evaluation metrics.
- Save trained models as MLflow artifacts.
- Perform a simple hyperparameter sweep.
- Compare multiple runs through the MLflow UI.
- Understand the role of experiment tracking in an MLOps workflow.

## 🔮 Future Enhancements

Possible extensions include:

- Integrate MLflow autologging.
- Add cross-validation to evaluate model stability.
- Implement automated best-run selection.
- Register the best-performing model in the MLflow Model Registry.
- Add model versioning and model serving.
- Automate training using GitHub Actions.
- Add a reproducible dependency file such as `requirements.txt`.
- Track experiments on a remote MLflow tracking server.

## 👨‍💻 Author

**Dhiraj Patil**

AI/ML Engineer | MLOps | Generative AI

GitHub: https://github.com/dhirajpatil-ai

## 📄 License

This project is available for educational and learning purposes. Add a `LICENSE` file if you want to specify formal terms for reuse and distribution.
