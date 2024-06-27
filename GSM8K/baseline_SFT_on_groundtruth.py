import argparse
import pickle
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments
import datasets
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score, classification_report
from peft import LoraConfig,PeftModel
import ast

import sys
sys.path.append('..')
from evaluation import evaluate_super_alignment

parser = argparse.ArgumentParser()

parser.add_argument(
    "--strong_model_name",
    type=str,
    choices=["Mistral-7B","LLAMA2-13B","LLAMA3-8B"],
)


def load_model(base_model_path, peft_model_path = None):    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
        device_map="auto",
        max_memory={0:"24GB",1:"24GB",2:"24GB",3:"24GB"},
        torch_dtype=torch.bfloat16,
        )
    if peft_model_path:
        model.load_adapter(peft_model_path)
    return model, tokenizer


def train_model(model, df_train, df_val, formatting_prompts_func, data_collator):

    ds_train = datasets.Dataset.from_pandas(df_train)
    ds_val = datasets.Dataset.from_pandas(df_val)

    training_args = TrainingArguments(
        output_dir='models/tmp',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=24,
        learning_rate=0.00005,
        logging_steps=100,
        remove_unused_columns=True,)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        max_seq_length=512,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    trainer.train()

    return trainer


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        label = example['label'][i]
        explanation = example['explanation'][i]
        prompt = f"### Instruction: The following is a grade-school level math question. \
In your response, provide the numerical answer in the first line, and then provide a brief explanation in the second line. \
Question: {example['question'][i]}. \
### Answer: {label} \n {explanation} "+strong_tokenizer.eos_token
        output_texts.append(prompt)
    print(output_texts[:10])
    return output_texts


if __name__ == "__main__":

    args = parser.parse_args()

    strong_model_name = args.strong_model_name

    # prepare dataset
    df_val = pd.read_csv('dataset/validation.csv')
    df_train, df_val = np.split(df_val, [int(.8*len(df_val))])
    print("# of training samples: ", len(df_train))
    print("# of validation samples: ", len(df_val))
    print(df_val.head())

    # load strong model
    if strong_model_name=="Mistral-7B":
        base_model_path = '../models/mistralai_Mistral-7B-Instruct-v0.2'
        strong_model, strong_tokenizer = load_model(base_model_path)
    elif strong_model_name=="LLAMA2-13B":
        base_model_path = "../models/Llama-2-13b-chat-hf"
        strong_model, strong_tokenizer = load_model(base_model_path)
    elif strong_model_name=="LLAMA3-8B":
        base_model_path = "../models/Llama-3-8B"
        strong_model, strong_tokenizer = load_model(base_model_path)
    print("Loaded strong model from: ", base_model_path)


    # data collator
    response_template = ". ### Answer:"
    response_template_ids = strong_tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=strong_tokenizer)

    # train model
    trainer = train_model(strong_model, df_train, df_val, formatting_prompts_func, collator)

    # save model
    trainer.save_model("models/{}-gsm8k-sft-on-groundtruth".format(strong_model_name))

    # evaluate model
    accuracy = evaluate_super_alignment(strong_model, strong_tokenizer, "GSM8K")

    # write to file
    with open("results/{}-gsm8k-sft-on-groundtruth.txt".format(strong_model_name), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.close()