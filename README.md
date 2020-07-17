# ecci_ml
Scripts covering (mostly) unsupervised attempts at finding and labelling dislocations in ECCI images of GaN.

The `datasets` directory containes archives that must be unpacked to the same directory before running any of the scripts. Upon first run of a training script two more directories will be created: `logs` for Tensorboard visualisation (use `tensorboard --logdir=./logs/[script prefix]`), and a directory containing the script prefix that will hold checkpoints and saved models for later reuse.

All scripts use click to enable command line functionality. Use the `--help` parameter for usage instructions.
