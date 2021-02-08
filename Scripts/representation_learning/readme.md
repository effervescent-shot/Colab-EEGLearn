# Representation Learning

this document is use to illustrate how to run the representation learning using our data.

For the representation learning, we train an encoder using the same networks that we used as decoding videos. The difference is that instead of predicting the labels, we predict the next frame (here we assume the video is 10 frames, so we predict the 11th frame).


- `run_RL.py` is used to run whole representation learning pipeline with leave-one-subejct-out cross-validation.

- `training_with_pytorch.py` contains all the grid-search training processing.

- `pytorch_models_RL.py` contains all types of models that we gonna use on representation learning. 

- `util_RL.py` are some helper functions for taking care of leave-one-subject-out cross-valdation piepline.

- `util_pytorch_RL.py` are some helper functions for building the representation learning pytorch models and training.


