# Train models for decoding videos

Scripts to decode directly videos in order to detect dreaming.

- `pytorch_models.py` define the models.

- `run_subject_based_validation.py` load the data, perform preprocessing steps and launch the leave-one-out cross validation grid search. 

- `training_with_pytorch.py` contains all the grid-search training process and the function to train a model for n epochs.

- `utils_pytorch.py` dataloader, optmizer and training(for one epoch only) are defined here.

- `utils_pytorch_SWA.py` same as above, but implementation fo stochastich weight averaging is available.

