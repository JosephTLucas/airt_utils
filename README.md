# airt_utils

For research-centric AI Red Teaming operations, there are best practices we can borrow from the evolution of MLOps. For example, both workflows involve multiple researchers focused on optimizations against open-ended research questions. At some point, the work from these researchers must be compared, potentially aggregated, and the best solution iterated upon. Often there is also a reporting phase where stakeholders need to be able to access R&D artifacts.

`airt_utils` is a collection of Python utilities for tracking AI Red Teaming experiments and operations, starting with the `airt_run` decorator that seamlessly integrates your functions with MLflow. It provides automatic logging of parameters, metrics, and artifacts, supporting both synchronous and asynchronous functions. This makes it ideal for a wide range of machine learning and data science workflows, particularly in the context of security research.

## Prerequisites

Before using `airt_utils`, you need to have MLflow running. MLflow is an open-source platform for managing the end-to-end machine learning lifecycle, including experimentation, reproducibility, and deployment.

To get started with MLflow:

1. Install MLflow:
   ```
   pip install mlflow
   ```

2. Start the MLflow tracking server:
   ```
   mlflow server --host 0.0.0.0 --port 5000
   ```

This will start the MLflow UI, which you can access at `http://localhost:5000`.

For more information on MLflow and its setup, refer to the [MLflow documentation](https://www.mlflow.org/docs/latest/index.html).

MLflow's purpose is to help manage the machine learning lifecycle, including experimentation, reproducibility, and deployment. In the context of AI security research, it can help track and compare different attack or defense strategies, model vulnerabilities, and the effectiveness of various countermeasures.

## Features

- 🚀 Easy integration with MLflow through the simple `airt_run` decorator
- 🔄 Support for both synchronous and asynchronous functions
- ⏱️ Optional timeout functionality
- 📊 Automatic logging of function parameters and return values
- 📁 Artifact logging support
- 🏷️ Custom tagging for runs
- 📈 Support for custom and system metrics
- 🔁 Retry mechanism for improved reliability
- 🛠️ Highly configurable to suit various use cases

## Installation

Install `airt_utils` using pip:

```bash
pip install airt_utils
```

## Quick Start

Here's an example of how to use the `airt_run` decorator in the context of AI security research:

```python
from airt_utils import airt_run
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

@airt_run(
    experiment_name="adversarial_example_generation",
    params=["epsilon", "n_samples"],
    artifacts=["adversarial_examples.npy"],
)
def generate_adversarial_examples(epsilon, n_samples):
    # Load a pre-trained model (assume we have a function for this)
    model = load_pretrained_model()
    
    # Generate benign samples (for simplicity, we'll use random data)
    X = np.random.rand(n_samples, 784)  # Assuming MNIST-like data
    y = np.random.randint(0, 10, n_samples)
    
    # Generate adversarial examples using Fast Gradient Sign Method (FGSM)
    X_adv = X + epsilon * np.sign(model.gradient(X, y))
    
    # Save adversarial examples
    np.save("adversarial_examples.npy", X_adv)
    
    # Evaluate model performance on adversarial examples
    y_pred = model.predict(X_adv)
    accuracy = accuracy_score(y, y_pred)
    
    return accuracy

result = generate_adversarial_examples(epsilon=0.1, n_samples=1000)
print(f"Model accuracy on adversarial examples: {result}")
```

This example will:
1. Create an MLflow experiment named "adversarial_example_generation"
2. Log the parameters `epsilon` and `n_samples`
3. Generate adversarial examples using the Fast Gradient Sign Method
4. Save the adversarial examples as an artifact
5. Evaluate and log the model's accuracy on these adversarial examples

## Advanced Usage

### Asynchronous Functions

`airt_run` supports async functions, which can be useful for distributed attacks or parallel processing:

```python
import asyncio
from airt_utils import airt_run

@airt_run(
    experiment_name="distributed_attack_simulation",
    timeout_seconds=300
)
async def distributed_attack():
    # Simulate multiple attack vectors
    tasks = [
        asyncio.create_task(sql_injection_attack()),
        asyncio.create_task(xss_attack()),
        asyncio.create_task(csrf_attack())
    ]
    results = await asyncio.gather(*tasks)
    return {
        "sql_injection_success": results[0],
        "xss_success": results[1],
        "csrf_success": results[2]
    }

result = asyncio.run(distributed_attack())
print(result)
```

### Custom Metrics

You can define custom metrics to track specific aspects of your security experiments:

```python
from airt_utils import airt_run

def attack_success_rate(result):
    return sum(result['success']) / len(result['success'])

@airt_run(
    experiment_name="model_robustness_evaluation",
    custom_metrics={"success_rate": attack_success_rate}
)
def evaluate_model_robustness(model, attack_samples):
    results = []
    for sample in attack_samples:
        prediction = model.predict(sample)
        results.append({
            "success": prediction != sample.true_label,
            "confidence": model.predict_proba(sample).max()
        })
    return results

attack_samples = load_attack_samples()  # Assume we have this function
model = load_target_model()  # Assume we have this function
results = evaluate_model_robustness(model, attack_samples)
```

This will log the success rate of the attacks as a custom metric.

## Configuration

`airt_run` offers various configuration options:

- `experiment_name`: Name of the MLflow experiment
- `params`: List of function parameters to log
- `artifacts`: List of files or directories to log as artifacts (protip: log entire configuration directories and debug logs for target applications)
- `tracking_uri`: MLflow tracking server URI
- `timeout_seconds`: Maximum execution time for the function
- `tags`: Dictionary of tags to apply to the run
- `log_system_metrics`: Whether to log system metrics
- `retry_attempts`: Number of retry attempts for MLflow operations
- `retry_delay`: Delay between retry attempts
- `custom_metrics`: Dictionary of custom metrics to compute and log

These configuration options allow you to tailor `airt_utils` to your specific AI security research needs, ensuring comprehensive tracking and analysis of your experiments.