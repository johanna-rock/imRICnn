from data_models.parameter_configuration import ParameterConfiguration


class EvaluationResult:

    def __init__(self, configuration: ParameterConfiguration, task_id, task, train_loss, val_loss, time_in_sec, snr, model_path):
        self.configuration = configuration
        self.task_id = task_id
        self.task = task
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.time_in_sec = time_in_sec
        self.snr = snr
        self.model_path = model_path
