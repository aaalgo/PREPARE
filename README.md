# A Solution to the Acoustic Track of the PREPARE Challenge

[Link to the Challenge](https://www.drivendata.org/competitions/299/competition-nih-alzheimers-acoustic-2/page/931/)

- `mental.py`: the core model.
- `train.py`: the training script.
- `extract.py`: the audio feature extraction script.
- `combine.py`: the script to combine prediction and generate submission.

See `solution.pdf` for details.  The code is copied from a working direction and needs to be modified to be runnable.  Apply `extract.py` to preprocess the audio file and generate the features in pickle format.  The training script does not save models, but instead saves the predictions.  The `combine.py` script combines the predictions and generates the submission file.