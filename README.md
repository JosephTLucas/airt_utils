# airt_utils

For research-centric AI Red Teaming operations, there are best practices we can borrow from the evolution of MLOps. For example, both workflows involve multiple researchers focused on optimizations against open-ended research questions. At some point, the work from these researchers must be compared, potentially aggregated, and the best solution iterated upon. Often there is also a reporting phase where stakeholders need to be able to access R&D artifacts.

`airt_utils` is a collection of Python utilities for tracking AI Red Teaming experiments and operations, starting with the `airt_run` decorator that seamlessly integrates your functions with MLflow. It provides automatic logging of parameters, metrics, and artifacts, supporting both synchronous and asynchronous functions. This makes it ideal for a wide range of machine learning and data science workflows.

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

Here's a simple example of how to use the `airt_run` decorator:

```python
from airt_utils import airt_run
import random

@airt_run(
    experiment_name="quick_start_example",
    params=["n"],
    artifacts=["output.txt"],
)
def generate_random_numbers(n):
    numbers = [random.randint(1, 100) for _ in range(n)]
    with open("output.txt", "w") as f:
        f.write(str(numbers))
    return sum(numbers)

result = generate_random_numbers(10)
print(f"Sum of generated numbers: {result}")
```

This example will:
1. Create an MLflow experiment named "quick_start_example"
2. Log the parameter `n`
3. Run the function and log its return value
4. Log the generated "output.txt" file as an artifact

## Advanced Usage

### Asynchronous Functions

`airt_run` supports async functions out of the box:

```python
import asyncio
from airt_utils import airt_run

@airt_run(
    experiment_name="async_example",
    timeout_seconds=5
)
async def async_operation():
    await asyncio.sleep(2)
    return "Async operation completed"

result = async_operation()
print(result)
```

### Custom Metrics

You can define custom metrics to be logged:

```python
from airt_utils import airt_run

def mean_metric(result):
    return sum(result) / len(result)

@airt_run(
    experiment_name="custom_metrics_example",
    custom_metrics={"mean": mean_metric}
)
def process_data(data):
    return [x * 2 for x in data]

result = process_data([1, 2, 3, 4, 5])
```

This will log the mean of the processed data as a custom metric.

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