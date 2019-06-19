
class ParameterConfiguration:
    def __init__(self,
                 input_window_size: int = None,
                 input_data_source=None,
                 optimization_algo=None,
                 criterion=None,
                 scheduler_partial=None,
                 hidden_act_func=None,
                 num_epochs=None,
                 num_model_initializations=None,
                 output_size=None,
                 scaler=None,
                 model=None,
                 num_hidden_units=None,
                 batch_size=None,
                 learning_rate=None,
                 input_size=None,
                 weight_decay=None,
                 dropout=None,
                 is_classification=None,
                 mat_path=None):
        self.input_window_size = input_window_size
        self.input_data_source = input_data_source
        self.model = model
        self.hidden_act_func = hidden_act_func
        self.optimization_algo = optimization_algo
        self.criterion = criterion
        self.scheduler_partial = scheduler_partial
        self.num_hidden_units = num_hidden_units
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_model_initializations = num_model_initializations
        self.output_size = output_size
        self.scaler = scaler
        self.input_size = input_size
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.is_classification = is_classification
        self.mat_path = mat_path

    def __repr__(self):
        r = '# Parameter Configuration #\n\n'
        r += 'Mat-Folder path: {}\n'.format(self.mat_path)
        r += 'Classification: {}\n'.format(self.is_classification)
        r += 'Input Window Size: {}\n'.format(self.input_window_size)
        r += 'Input Data Source: {}\n'.format(self.input_data_source.__name__)
        if self.scaler:
            try:
                r += 'Scaler: {}\n'.format(self.scaler.__name__)
            except AttributeError:
                r += 'Scaler: {}\n'.format(self.scaler)
        else:
            r += 'Scaler: None\n'
        try:
            r += 'Activation func: {}\n'.format(self.hidden_act_func.__name__)
        except AttributeError:
            r += 'Activation func: {}\n'.format(self.hidden_act_func)
        r += 'Hidden units: {}\n'.format(self.num_hidden_units)
        r += 'Dropout probabilities: {}\n'.format(self.dropout)
        r += 'Batch size: {}\n'.format(self.batch_size)
        r += 'Learning rate: {}\n'.format(self.learning_rate)
        r += 'Weight decay: {}\n'.format(self.weight_decay)
        r += 'Criterion: {}\n'.format(self.criterion)
        try:
            r += 'Optimizer: {}\n'.format(self.optimization_algo.__name__)
        except AttributeError:
            r += 'Optimizer: {}\n'.format(self.optimization_algo)
        try:
            r += 'Scheduler: {}\n'.format(self.scheduler_partial.func.__name__)
        except AttributeError:
            r += 'Scheduler: {}\n'.format(self.scheduler_partial)
        r += 'Num model parameters: {}\n'.format(count_parameters(self.model))
        r += 'Model: {}\n'.format(self.model)
        return r


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
