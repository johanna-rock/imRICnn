# Complex Signal Denoising and Interference Mitigation for Automotive Radar Using Convolutional Neural Networks #

This repository contains the code for training and evaluating CNNs for automotive radar signal denoising as introduced in the following paper:
**Complex Signal Denoising and Interference Mitigation for Automotive Radar Using Convolutional Neural Networks (FUSION 2019)**

## Citation
If you find this approach useful in your research, please consider citing:
```
@INPROCEEDINGS{Rock1907:Complex,
    AUTHOR="Johanna Rock and Mate Toth and Elmar Messner and Paul Meissner and Franz Pernkopf",
    TITLE="Complex Signal Denoising and Interference Mitigation for Automotive Radar Using Convolutional Neural Networks",
    BOOKTITLE="2019 22nd International Conference on Information Fusion (FUSION) (FUSION 2019)",
    YEAR=2019
}
```

## Usage

1. Clone this repository: `git clone https://github.com/johanna-rock/im_ricnn.git`
2. Create a virtual environment from the included environment.yml and activate it
    a. Create using conda: `conda env create -f environment.yml`
    b. Activate using conda: `conda activate im-cnn-env`
3. Download simulated sample data from [https://cloud.tugraz.at/index.php/s/gWpr5RfzKBdbAaW](https://cloud.tugraz.at/index.php/s/gWpr5RfzKBdbAaW), unzip the file and save it to `data/radar-data`.

## Training
Run `run_scripts/run_training.py` to train and evaluate a CNN with the specified configuration.

## Evaluation
Run `run_scripts/run_evaluation.py` to evaluate a pre-trained model with the specified configuration.
