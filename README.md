# Studying human-level cognition with neural networks trained for many tasks

This work studies how neural networks can perform a large range of cognitive
tasks common in human cognitive neuroscience and psychology studies.

# Getting started

You need to have the following libraries installed:
- numpy
- tensorflow. Tested with 1.8, other versions will probably work as well.
- opencv 3. Tested with 3.4.1.15. Other 3.* version should work as well.

```
 pip install opencv-python tensorflow opencv-python
```

Some visualizations require matplotlib.

You can use your favorite python IDE to import the project. Alternatively, you can add the base directory to you PYTHONPATH.


# Datasets

We have pre-generated two variants of the COG dataset:
- a canonical one with parameters (4, 3, 1) - 4 frames, maximum memory duration of 3, maximum distractors of 1
- a "hard" one with parameters (8, 7, 10).

These datasets can be downloaded at XXX

Dataset stats:
- Examples per task family for training/validation/testing: 227280/11364/11364
- Task families: 44
- Total examples for training/validation/testing: 10000320/500016/500016
- gzip'ed canonical dataset is 1.2GB
- gzip'ed hard dataset is 5.0GB


These datasets were generated with:
```
python cognitive/generate_dataset.py --examples_per_family=227280 --parallel=12 --output_dir=/tmp/cog/train_data_4_3_1_10M/ --epochs=4 --max_memory=3 --max_distractors=1

python cognitive/generate_dataset.py --examples_per_family=227280 --parallel=12 --output_dir=/tmp/cog/train_data_8_7_10_10M/ --epochs=8 --max_memory=7 --max_distractors=10
```

You can generate other variants using similar command lines. Most flags are self-explanatory. The 'parallel' flag instructs the script to generate the dataset in parallel using the provided number of processes. By default dataset files are gzip'ed. If you don't want to gzip automatically, use '--compress=false' flag.

## Dataset format
Data is stored in text files. Each line of a file contains serialzied json string. An examples of such a string (pretty printed) is included below.

```
{
    "answers": [
        [
            0.666,
            0.196
        ],
        [
            0.16,
            0.299
        ]
    ],
    "epochs": 3,
    "family": "Go",
    "objects": [
        {
            "color": "blue",
            "epochs": 0,    // can be an int or a list of ints
            "is_distractor": false,
            "location": [
                0.666,
                0.196
            ],
            "shape": "vbar"
        },
        {
        {
            "color": "grey",
            "epochs": [1, 2],
            "is_distractor": true,
            "location": [
                0.203,
                0.414
            ],
            "shape": "a"
        }
    ],
    "question": "point now blue m"
}
```

Describe the format ... XXX


# Training

You can train a network locally with default hyperparameters by running
```python
python cognitive/train.py --data_dir=<directory with train/val/test data> \
     --train_dir=<directory to use for checkpoints, tensorboard summaries, and hparam dump>
```

Describe how to train with data on the fly... XXX

Describe how to generate CLEVR data and train on it... XXX

# Disclaimer

This is not an officially supported Google product.
