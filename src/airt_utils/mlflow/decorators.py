import mlflow
import logging
import asyncio
from pathlib import Path
from functools import wraps
from typing import Callable, List, Union, Optional, Dict, Any
from inspect import signature
import concurrent.futures
import time

class AIRTMLflowError(Exception):
    """Custom exception for AIRT MLflow errors."""
    pass

def airt_run(
    experiment_name: Optional[str] = None,
    params: List[str] = [],
    artifacts: List[Union[str, Path]] = [],
    tracking_uri: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    tags: Dict[str, str] = {},
    log_system_metrics: bool = True,
    retry_attempts: int = 1,
    retry_delay: float = 1.0,
    custom_metrics: Dict[str, Callable] = {},
):
    """
    A decorator to log function execution details to MLflow.

    Args:
        experiment_name (Optional[str]): The name of the MLflow experiment. If None, uses the active experiment.
        params (List[str]): List of parameter names to log.
        artifacts (List[Union[str, Path]]): List of file or directory paths to artifacts to log.
        tracking_uri (Optional[str]): The URI of the MLflow tracking server. If None, uses the current tracking URI.
        timeout_seconds (Optional[int]): The maximum number of seconds the function is allowed to run.
        tags (Dict[str, str]): Additional tags to log with the run.
        log_system_metrics (bool): Whether to log system metrics.
        retry_attempts (int): Number of retry attempts for MLflow operations.
        retry_delay (float): Delay between retry attempts in seconds.
        custom_metrics (Dict[str, Callable]): Custom metrics to log, where keys are metric names and values are functions to compute the metric.

    Returns:
        Callable: The wrapped function with MLflow logging and asyncio timeout.

    Raises:
        AIRTMLflowError: If there's an error in MLflow logging or execution.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def run_mlflow_operations():
                if tracking_uri:
                    mlflow.set_tracking_uri(uri=tracking_uri)

                if experiment_name:
                    mlflow.set_experiment(experiment_name)

                with mlflow.start_run(log_system_metrics=log_system_metrics) as run:
                    if tags:
                        mlflow.set_tags(tags)

                    sig = signature(func)
                    bound_args = sig.bind_partial(*args, **kwargs)
                    bound_args.apply_defaults()
                    all_kwargs = bound_args.arguments
                    self_or_cls = all_kwargs.pop('self', None) or all_kwargs.pop('cls', None)

                    if params:
                        param_dict = {param: all_kwargs.get(param) for param in params}
                        mlflow.log_params(param_dict)

                    result = None
                    try:
                        if asyncio.iscoroutinefunction(func):
                            loop = asyncio.get_event_loop()
                            task = loop.create_task(func(self_or_cls, **all_kwargs) if self_or_cls else func(**all_kwargs))
                            result = loop.run_until_complete(asyncio.wait_for(task, timeout=timeout_seconds))
                        else:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(func, self_or_cls, **all_kwargs) if self_or_cls else executor.submit(func, **all_kwargs)
                                result = future.result(timeout=timeout_seconds)
                    except asyncio.TimeoutError:
                        raise AIRTMLflowError(f"Function {func.__name__} timed out after {timeout_seconds} seconds.")
                    except Exception as e:
                        raise AIRTMLflowError(f"Function {func.__name__} failed with error: {str(e)}")
                    finally:
                        if result is not None:
                            if isinstance(result, dict):
                                mlflow.log_params(result)
                            else:
                                mlflow.log_metric('return_value', result if isinstance(result, (int, float)) else 0)

                        for artifact in artifacts:
                            artifact_path = all_kwargs.get(artifact, None) or Path(artifact)
                            artifact_path = Path(str(artifact_path))

                            if artifact_path.exists():
                                if artifact_path.is_file():
                                    mlflow.log_artifact(str(artifact_path))
                                elif artifact_path.is_dir():
                                    mlflow.log_artifacts(str(artifact_path))
                            else:
                                logging.warning(f"Artifact {artifact_path} does not exist and will not be logged.")

                        for metric_name, metric_func in custom_metrics.items():
                            try:
                                metric_value = metric_func(result)
                                mlflow.log_metric(metric_name, metric_value)
                            except Exception as e:
                                logging.error(f"Error computing custom metric {metric_name}: {str(e)}")

                    return result

            for attempt in range(retry_attempts):
                try:
                    return run_mlflow_operations()
                except Exception as e:
                    if attempt < retry_attempts - 1:
                        logging.warning(f"MLflow operation failed. Retrying in {retry_delay} seconds. Error: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        raise AIRTMLflowError(f"MLflow operation failed after {retry_attempts} attempts. Error: {str(e)}")

        return wrapper
    return decorator