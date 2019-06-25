# Complex Signal Denoising and Interference Mitigation for Automotive Radar Using Convolutional Neural Networks #

This repository contains the code for training and evaluating CNNs for automotive radar signal denoising as introduced in the paper [**Complex Signal Denoising and Interference Mitigation for Automotive Radar Using Convolutional Neural Networks**](https://arxiv.org/abs/1906.10044)

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
	1. Create using conda: `conda env create -f environment.yml`
	2. Activate using conda: `conda activate im-cnn-env`
3. Set the python path using `export PYTHONPATH="/path/to/imRICnn"`
4. Download simulated sample data from [https://cloud.tugraz.at/index.php/s/gWpr5RfzKBdbAaW](https://cloud.tugraz.at/index.php/s/gWpr5RfzKBdbAaW), unzip the file and save it to `imRICnn/data/radar-data`.

## Training
Run `python -m run_scripts.run_training.py` to train and evaluate a CNN with the configuration specified in run_training.py.

## Evaluation
Run `python -m run_scripts.run_evaluation.py` to evaluate a pre-trained model with the configuration specified in run_evaluation.py.