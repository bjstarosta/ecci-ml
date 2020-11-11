# Machine learning assisted detection of threading dislocations in ECC images

## Install

Use `env.txt` to replicate the Anaconda environment using:
```
$ conda create --name <env_name> --file env.txt
```

Once you activate the environment, you may want to (re)generate some of the synthetic datasets used. Navigate to the `datasets/` directory and use the bash scripts within to do this. In each dataset directory there are two bash scripts:
* `gen.sh` - generates new data based on random variables
* `regen.sh` - regenerates data in the exact same state as was used for training

Look at the bash scripts and `semgen/semgen.py --help` for documentation of the commands used.

## Usage

All scripts use [Click](https://github.com/pallets/click) to enable command line functionality. Use the `--help` parameter for usage instructions.

**Entry points:**
```
$ python train.py
$ python predict.py
```

## License

[GNU GPL v.3](https://github.com/bjstarosta/ecci_ml/blob/master/LICENSE)
