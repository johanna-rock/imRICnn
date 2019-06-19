from run_scripts import verbose


class EarlyStopping:
    def __init__(self, steps_to_wait, epsilon=0):  # epsilon=10**-20
        assert(steps_to_wait >= 0)
        self.steps_to_wait = steps_to_wait
        self.epsilon = epsilon
        self.best_loss = float('inf')
        self.waited_steps = 0

    def should_stop(self, current_loss, current_step):
        if self.best_loss - current_loss > self.epsilon:
            self.best_loss = current_loss
            self.waited_steps = 0
        elif self.waited_steps < self.steps_to_wait:
            self.waited_steps += 1
        else:
            if verbose:
                print('Early stopping at time step {} after waiting {} steps'.format(current_step, self.steps_to_wait))
            return True
        return False
