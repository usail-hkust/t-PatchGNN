# t-PatchGNN

This is an official implementation of [*Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach*](https://openreview.net/pdf?id=UZlMXUGI6e).

## Requirements

t-PatchGNN has been tested using Python 3.9.

To have consistent libraries and their versions, you can install needed dependencies for this project running the following command:

```shell
pip install -r requirements.txt
```

## Datasets

We use four datasets to benchmark our method. 

For Physionet and Human Activity, our code will automatically download the raw data and preprocess them.

For USHCN, following the [GRU-ODE-Bayes](https://github.com/edebrouwer/gru_ode_bayes/tree/master), we use the same preprocessed data `small_chunked_sporadic.csv` as the raw data.

For MIMIC, because of the [PhysioNet Credentialed Health Data License](https://physionet.org/content/mimiciii/view-dua/1.4/), you need to first request the raw database from [here](https://physionet.org/content/mimiciii/1.4/). The database version we used here is v1.4. After downloading the raw data, following the preprocessing of [Neural Flows](https://github.com/mbilos/neural-flows-experiments/blob/master/nfe/experiments/gru_ode_bayes/data_preproc/mimic_prep.ipynb), you will finally get the `full_dataset.csv` which is used as the raw data in our experiment.

## Run the Model

For the specific configurations utilized to obtain the principal experimental outcomes presented in the paper, kindly refer to the script "run_all.sh". To replicate these results, please execute the below command.

```shell
sh ./tPatchGNN/scripts/run_all.sh
```

Example:

```shell
python run_models.py \
    --dataset {dataset} --state {def_or_debug} --history {length_of_observation}\
    --patience {patience_of_earlystopping} --batch_size {batch_size} --lr {learning_rate} \
    --patch_size {window_size_for_a_patch} \
    --stride {period_stride_for_patch_sliding} \
    --nhead {heads_in_Transformer} \
    --tf_layer{number_of_layer_in_Transformer} \
    --nlayer {num_of_layer_in_Time_Series_Model} \
    --te_dim {number_of_units_for_time_encoding} \
    --node_dim {number_of_units_for_node_vectors} \
    --hid_dim {number_of_units_per_hidden_layer} \
    --outlayer {model_for_outlayer} --seed {seed} --gpu {gpu}
```

- `dataset`: the dataset name, select from `[physionet, mimic, activity, ushcn]`.
- `seed`: the seed for parameter initialization.
- `history`: the length of the time for observation, the rest will be used for forecasting. Please note that different datasets have varying time spans and levels of granularity.

## Citation

```shell
@inproceedings{zhangirregular,
  title={Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach},
  author={Zhang, Weijia and Yin, Chenlong and Liu, Hao and Zhou, Xiaofang and Xiong, Hui},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
