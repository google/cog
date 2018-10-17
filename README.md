# A dataset and architecture for visual reasoning with a working memory
This is the code accompanying ECCV 2018 paper https://arxiv.org/abs/1803.06092 .


# Getting started

You need to have the following libraries installed:
- numpy
- tensorflow. Tested with 1.8, other versions will probably work as well.
- opencv 3. Tested with 3.4.1.15. Other 3.* version should work as well.
- scipy
```
 pip install opencv-python tensorflow opencv-python scipy
```

Some visualizations require matplotlib.

You can use your favorite python IDE to import the project. Alternatively, you can add the base directory to your PYTHONPATH.


# Datasets

We have pre-generated two variants of the COG dataset:
- a canonical one with parameters (4, 3, 1) - 4 frames, maximum memory duration of 3, maximum distractors of 1
- a "hard" one with parameters (8, 7, 10).

These datasets can be downloaded at:
- canonical: https://storage.googleapis.com/cog-datasets/data_4_3_1.tar (md5sum 7a03bae2e3f31e3309b1d0e5b00f46aa)
- hard: https://storage.googleapis.com/cog-datasets/data_8_7_10.tar (md5sum 9a766fd40732376833ec743e0792437e)


Dataset stats:
- Examples per task family for training/validation/testing: 227280/11364/11364
- Task families: 44
- Total examples for training/validation/testing: 10000320/500016/500016
- gzip'ed canonical dataset is 1.2GB
- gzip'ed hard dataset is 5.0GB


These datasets were generated with:
```
python cognitive/generate_dataset.py --examples_per_family=227280 \
  --parallel=12 --output_dir=/tmp/cog/data_4_3_1 \
  --epochs=4 --max_memory=3 --max_distractors=1

python cognitive/generate_dataset.py --examples_per_family=227280 \
  --parallel=12 --output_dir=/tmp/cog/data_8_7_10 \
  --epochs=8 --max_memory=7 --max_distractors=10
```

You can generate other variants using similar command lines.
The `--parallel` flag instructs the script to generate the dataset in parallel using
the provided number of processes. By default dataset files are gzip'ed.
If you don't want to gzip automatically, use `--compress=false` flag.
Use `-h` to see all the options.

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
    "epochs": 2,
    "family": "Go",
    "objects": [
        {
            "color": "blue",
            "epochs": 0,
            "is_distractor": false,
            "location": [
                0.666,
                0.196
            ],
            "shape": "vbar"
        },
        # < more more objects>
        {
            "color": "grey",
            "epochs": [0, 1],
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

The fields in each example mean the following:
- **epochs**: the number of epochs (aka "frames") in this example.
- **answers**: a list of strings or a list of floating point coordinates. The length
of the list is equal to the number of epochs. The string values include a special
value 'invalid' which means that the question has no valid answer after the corresponding epoch.
- **family**: the task family this example belongs to.
- **question**: the question that is asked about this example (aka "rule").
- **objects**: a list of objects that are part of this example. Each objects has the following fields:
  - **color**: color of the object. The RGB values for each colors can be found
      in cognitive/constants.py
  - **shape**: a description of the shape. Shapes can be standard ascii letters as well as
      custom shapes like "vbar" meaning "vertical bar".
  - **location**: a pair of floating point numbers between 0 and 1.
  - **epochs**: an integer or a list of integers identifying the epochs (starting with 0) that
      this object is present in.
  - **is_distractor**: a boolean specifying if this object is used in the answering the question.


# Training

## COG

You can train a network locally with default hyperparameters by running
```
python cognitive/train.py --data_dir=<directory with train/val/test data> \
     --train_dir=<directory to use for checkpoints, summaries, etc>
```

If no `data_dir` is given `train.py` will generate examples for training and validation on the fly.

When training on a canonical dataset on GPU, you should see the results of intermediary evaluation every 10-20 minutes. You can use `--display_step` to perform evaluation more often (default value is 3000 batches). Use `-h` to see all the options. The default hyper-parameter values are set in the `get_default_hparams_dict` function.

## CLEVR
To train in CLEVR, first, you need to download the CLEVR dataset and unzip it.
Then, convert it to TFRecord format using the following command line:
```
python clevr/data_converter.py --command=convert --data_type='all' \
  --tfrecord_dir=<directory for output tfrecord files> \
  --raw_clevr_dir=<directory containing extracted CLEVR files>
```
Conversion can take about 20min.

To train on CLEVR dataset use the following command
```
python clevr/train.py --data_dir=<directory containing converted tf records> \
     --train_dir=<directory to use for checkpoints, summaries, etc>
```

To produce CLEVR test set outputs, run:
```
python clevr/train.py --data_dir=<directory containing converted tf records> \
     --train_dir=<directory containing checkpoint to load> \
     --clevr_test_output=<directory to output test results>
```

# Testing

## COG

We have included a `cognitive/test.py` script to evaluate a model on the test dataset
as well as a model checkpoint trained on canonical cog that reaches 94% accuracy.

You can run this model on examples generated on the fly with:
```
python cognitive/test.py --model_dir=./trained/cog_canonical
```

To run on a saved dataset, e.g. a downloaded canonical dataset, use the `--data_dir` option:
```
python cognitive/test.py --model_dir=./trained/cog_canonical --data_dir=/tmp/data_4_3_1/
```


# Disclaimer

This is not an officially supported Google product.
