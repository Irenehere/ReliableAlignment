import argparse
import pickle
import numpy as np
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
    "--weak_model_name",
    type=str,
    choices=["Mistral-7B","LLAMA2-7B"],
)


def load_model(base_model_path, peft_model_path = None):    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model_path,
        device_map="auto",
        max_memory={0:"24GB", 1:"24GB",2:"24GB",3:"24GB"},
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


if __name__ == "__main__":

    args = parser.parse_args()
    
    weak_model_name = args.weak_model_name

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['ctx'])):
            answer = "ABCD"[int(example['label'][i])]
            endings_choices = ast.literal_eval(example['endings'][i])
            prompt = f"### Instruction: Choose an appropriate ending for the provided sentence based on your commonsense knowledge.\
In your response, choose an answer from A,B,C,D. \
Sentence: {example['ctx'][i]}. \
Choices: A. {endings_choices[0]} B. {endings_choices[1]} C. {endings_choices[2]} D. {endings_choices[3]}. \
### Answer: {answer}. "+endings_choices[int(example['label'][i])]+tokenizer.eos_token
            output_texts.append(prompt)
        print(output_texts[:10])
        return output_texts

    # prepare dataset
    df_train = pd.read_csv('dataset/train.csv')
    df_train, df_val = np.split(df_train, [int(.8*len(df_train))])
    print("# of training samples: ", len(df_train))
    print("# of validation samples: ", len(df_val))
    print(df_train.head())    

    # load model
    if weak_model_name == "Mistral-7B":
        model_path = "../models/mistralai_Mistral-7B-Instruct-v0.2"
    elif weak_model_name == "LLAMA2-7B":
        model_path = "../models/Llama-2-7b-chat-hf"
    else:
        raise ValueError("Invalid model name")
    
    print("Loading model from: ", model_path)
    model, tokenizer = load_model(model_path)

    # data collator
    response_template = ". ### Answer:"
    response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    # train model
    trainer = train_model(model, df_train, df_val, formatting_prompts_func, collator)

    # save model
    trainer.save_model("models/{}-hellaswag-train-sft".format(weak_model_name))

    # evaluate model
    accuracy, report = evaluate_super_alignment(model, tokenizer, "Hellaswag")

    # write to file
    with open("results/{}_weak_supervisor_performance.txt".format(weak_model_name), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.write(report)
        f.close()