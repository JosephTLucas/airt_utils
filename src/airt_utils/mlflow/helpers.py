import mlflow
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from typing import Dict, Optional
import requests

def check_mlflow_connectivity(tracking_uri: Optional[str] = None) -> bool:
    """
    Check if the MLflow tracking server is accessible.

    Args:
        tracking_uri (Optional[str]): The URI of the MLflow tracking server. If None, uses the current tracking URI.

    Returns:
        bool: True if the server is accessible, False otherwise.
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.list_experiments()
        return True
    except Exception as e:
        print(f"Failed to connect to MLflow server: {str(e)}")
        return False

def init_mlflow_experiment(
    experiment_name: str,
    user_permissions: Dict[str, str] = {},
    tracking_uri: Optional[str] = None
) -> Optional[Experiment]:
    """
    Initialize or get an MLflow experiment and set user permissions.

    Args:
        experiment_name (str): The name of the MLflow experiment.
        user_permissions (Dict[str, str]): Dictionary of usernames and their permission levels ('READ', 'EDIT', or 'MANAGE').
        tracking_uri (Optional[str]): The URI of the MLflow tracking server. If None, uses the current tracking URI.

    Returns:
        Optional[Experiment]: The MLflow Experiment object, or None if initialization fails.

    Raises:
        MlflowException: If there's an error in creating or retrieving the experiment.
    """
    if not check_mlflow_connectivity(tracking_uri):
        print("MLflow server is not accessible. Aborting experiment initialization.")
        return None

    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name=experiment_name)
            experiment = mlflow.get_experiment(experiment_id)
            print(f"Created new experiment: {experiment_name}")
        else:
            print(f"Using existing experiment: {experiment_name}")

        for user, permission in user_permissions.items():
            if permission not in ['READ', 'EDIT', 'MANAGE']:
                print(f"Invalid permission '{permission}' for user {user}. Skipping.")
                continue
            try:
                mlflow.set_experiment_permission(
                    experiment.experiment_id,
                    user,
                    permission=permission
                )
                print(f"Set {permission} permission for user {user} on the experiment.")
            except MlflowException as e:
                print(f"Failed to set {permission} permission for user {user}: {str(e)}")

        return experiment

    except MlflowException as e:
        raise MlflowException(f"Error initializing MLflow experiment: {str(e)}")