from . import tuner
from common import api
from common import status_type
import logger
LOGGER = logger.getlogger("controller.experiments")


def get_example_v1beta1(client, namespace, name, algorithm, repeat, example):
    # check example
    example_entry, _ = api.list_example_version(client)
    if not example in example_entry:
        return "Not found example"
    
    algorithm = api.absolute_algorithm(algorithm)
    
    # make experiment names
    experiment_names = []
    for index in range(0, int(repeat)):
        experiment_names.append(f"{name}-{example}-{algorithm}-{index+1}")
        
    LOGGER.info(f"experiment_names={experiment_names}")
    
    
     # deploy example experiment
    status = []
    for experiment in experiment_names:
        spec= tuner.build_example_v1beta1(namespace, experiment, algorithm)
        status.append(api.create_example_experiment(client, namespace, spec))
        
    if status_type.ExperimentFailed in list(set(status)):
        return status_type.ExperimentFailed
    return status_type.ExperimentSucceeded