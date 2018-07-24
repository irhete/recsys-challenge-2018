# RecSys Challenge 2018 - Team Latte


The 2018 RecSys challenge was organized by Spotify. In the challenge, participants were challenged to create a recommendation system for music tracks, able to perform automatic playlist continuation. A dataset containing one million playlists was made available, and the teams could participate in two distinct tracks: The main track where only the provided Million Playlist Dataset is used, and the creative track where additional public data could be used.

For more information, please refer to:

* http://www.recsyschallenge.com/2018/
* https://recsys-challenge.spotify.com/


Our team focused in the main track of the challenge. This repository contains the scripts used to generate the final solution.

Team members: [Irene Teinemaa](https://irhete.github.io), [Niek Tax](https://scholar.google.com.au/citations?user=XkRvCC4AAAAJ&hl=en&oi=ao), [Carlos Bentes](https://www.cbentes.com/), Maksym Semikin, Meri Liis Treimann, Christian Safka.


## Generate Submission


* Step 0: Create a new Conda environment and install the required libraries

```shell

create conda --name recsys python=3.6
source activate recsys
pip install -r requirements.txt
```

* Step 1: Generate Optimization and Validation subsets from train data

```shell

python generate_opt_val_data.py
```

* Step 2: Create intermediary structures 


```shell

python create_pickle_files.py
```


* Step 3: The model training and submission file is generated with:

```shell

python train_model_and_generate_submission.py submission 1000000
```

This script contains optimal parameters hardcoded to best combine models. These parameters calculation is described in next section.


## Model Optimization (Optional)

The train_model_and_generate_submission.py script contains all parameters used in the final submission. These parameters were determined using a Tree-structured Parzen Estimator (TPE) with the following command:


```shell

python optimize_parameters_1.py playlist_count n_top_songs n_opt_iter sample_group
```

where:

* playlist_count: The number of playlists to be included, example: 400000
* n_top_songs: The number of most frequent songs to be included, example: 100000
* n_opt_iter: The number of iteration rounds, example: 100
* sample_group: The target group for optimization, from: "title_only", "1_with_title", "5_no_title", "5_with_title", "10_no_title", "10_with_title", "25_first", "25_random", "100_first", "100_random".


The output of the optimize_parameters_1 script for every sample_group is hardcoded in the optimize_parameters_2, and the final set of parameters are calculated using Grid Search with the following command:

```shell

python optimize_parameters_2.py playlist_count n_top_songs sample_group
```

where:

* playlist_count: The number of playlists to be included, example: 400000
* n_top_songs: The number of most frequent songs to be included, example: 100000
* sample_group: The target group for optimization, from: "title_only", "1_with_title", "5_no_title", "5_with_title", "10_no_title", "10_with_title", "25_first", "25_random", "100_first", "100_random".


## Model Validation (Optional)

The model was validated using 10k playlists generated in Step 1. The output of the validation step can be obtained with:


```shell

python train_model_and_generate_submission.py val 990000

python validate.py validation.csv
```
