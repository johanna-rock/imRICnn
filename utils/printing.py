from run_scripts import print_, JOB_DIR, task_id


class PrintColors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'


def print_evaluation_summary(time_elapsed, phase, snrs=None, snr_labels=None, accuracy=None):
    print_()
    print_('{:<24}: {}'.format('data', phase))
    print_('{:<24}: {:.0f}m {:.0f}s'.format('duration', time_elapsed // 60, time_elapsed % 60))
    if snrs is not None:
        for i in range(len(snrs)):
            print_('{:<24}: {:>22.10f}'.format(snr_labels[i], snrs[i]))
    if accuracy is not None:
        print_('{:<24}: {:>22.10f}'.format('accuracy', accuracy))
    print_()
