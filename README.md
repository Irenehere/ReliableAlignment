
# Reliability-Aware Alignment

Codes to the paper Improving Weak-to-Strong Generalization with Reliability-Aware Alignment


## Requirments

Please create the environment from the environment.yml file:

```conda env create -f environment.yml```

Then enter your conda environment to run the codes.


## Main Experiments

We did experiments on four datasets: [Hellaswag](https://huggingface.co/datasets/Rowan/hellaswag), [MMLU](https://huggingface.co/datasets/cais/mmlu), [ETHICS (commonsense subset)](https://github.com/hendrycks/ethics), and [GSM8K](https://huggingface.co/datasets/openai/gsm8k). To reproduce the results of each dataset, please enter the corresponding dataset directory.

Besides, to run the codes, please download your own pre-trained models into the ```models``` directory and name them like 
```
models
└───Llama-2-7b-chat-hf
 │   files of the model and tokenizer
└───Llama-2-13b-chat-hf
 │   files of the model and tokenizer
└───Llama-3-8B
 │   files of the model and tokenizer
└───mistralai_Mistral-7B-Instruct-v0.2
 │   files of the model and tokenizer
```

### Create weak supervisors

The first step to conducting weak-to-strong generalization is to train a weak supervisor on the training data. To run the codes that create weak supervisor, go to the corresponding dataset directory and first create two folders to store the trained models and results by ```mkdir models results```, then run

```python create_weak_supervisor_on_{dataset_name}.py --weak_model_name {weak_model_name}```,

where the ```{weak_model_name}``` should be replaced by choosing from ["Mistral-7B","LLAMA2-7B"].


### Baseline: naive weak-to-strong generalization

In the naive weak-to-strong generalization setting, we prompt the weak labels from the weak supervisor and SFT the strong model using the naive weak labels. To reproduce the results, go to the corresponding dataset directory and run

```python baseline_SFT_weak_supervision.py  --weak_model_name {weak_model_name} --strong_model_name {strong_model_name}```,

where the ```{weak_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-7B"], and  ```{strong_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-13B","LLAMA3-8B"].  

### Ceiling: SFT the strong model on ground truth

The ceiling performance is achieved by fine-tuning the strong model on the ground truth label of the validation data. To run the code, use

```python baseline_SFT_on_groundtruth.py --strong_model_name {strong_model_name}```,

where the ```{strong_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-13B","LLAMA3-8B"].

### Our method 1: Uncertainty Filtering

In this method, we prompt a question with multiple variations to the model, which returns multiple answers from the model. We then compute the uncertainty of these answers using the entropy score. Answers with high entropy are filtered out, and only the low-entropy answers are preserved for model SFT. Run

```python our_method_uncertainty_filtering.py  --weak_model_name {weak_model_name} --strong_model_name {strong_model_name}```,
 
or 

```python our_method_uncertainty_filtering.py  --weak_model_name {weak_model_name} --strong_model_name {strong_model_name} --do_sampling``` for a sampled dataset.
 

The ```{weak_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-7B"], and  ```{strong_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-13B","LLAMA3-8B"].  

### Our method 2: Reliability Reweighting

In this method, we use the multiple answers from the previous prompting. Then we reweight the loss of (question, weak answer) sample with the reliability score of the weak answer. 

Note that you should run method 1 first before running the script for method 2. This is because method 2 re-uses the file {dataset}_validation_with_entropy_from_{weak_model_name}.csv generated from method 1. To reproduce the results of method 2, run

```python our_method_reweighting.py  --weak_model_name {weak_model_name} --strong_model_name {strong_model_name}```,

or 

```python our_method_reweighting.py  --weak_model_name {weak_model_name} --strong_model_name {strong_model_name} --do_sampling``` for a sampled dataset.


where the ```{weak_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-7B"], and  ```{strong_model_name}``` should be chosen from ["Mistral-7B","LLAMA2-13B","LLAMA3-8B"].  



## Visualization

To reproduce Figure 3 and 5 from the paper, please use the ```Visualization_on_entropy_uncertainty.ipynb``` notebook.

To reproduce Figure 4 and 6 from the paper, please use the ```Visualization_on_probability_reliability_score.ipynb``` notebook.
