import argparse
import pickle
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, AutoModelForSequenceClassification, Trainer
import datasets
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score, classification_report
from peft import LoraConfig,PeftModel
from scipy.stats import entropy
import ast
import os
import sys
sys.path.append('..')
from evaluation import evaluate_super_alignment


parser = argparse.ArgumentParser()

parser.add_argument(
    "--strong_model_name",
    type=str,
    choices=["Mistral-7B","LLAMA2-13B","LLAMA3-8B"],
)

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
        max_memory={0:"24GB",1:"24GB",2:"24GB",3:"24GB"},
        torch_dtype=torch.bfloat16,
        )
    if peft_model_path:
        model.load_adapter(peft_model_path)
    return model, tokenizer


def prompt_model(prompt, model, tokenizer, max_new_tokens=1):
    inputs =  tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
        outputs_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return outputs_string


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


def get_labels_from_weak_model(validation_df, model, tokenizer):
    weak_labels = []
    for row in tqdm(validation_df.iterrows(), total=len(validation_df)):
        endings_choices = ast.literal_eval(row[1]['endings'])
        prompt = f"### Instruction: Choose an appropriate ending for the provided sentence based on your commonsense knowledge.\
In your response, choose an answer from A,B,C,D, and provide a brief explanation. \
Sentence: {row[1]['ctx']}. \
Choices: A. {endings_choices[0]} B. {endings_choices[1]} C. {endings_choices[2]} D. {endings_choices[3]}. \
### Answer: "
        model_answer = prompt_model(prompt, model, tokenizer,max_new_tokens=50)
        model_answer = model_answer.split("### Answer:")[1].strip().strip("_- ")
        weak_labels.append(model_answer)
        if row[0]%20==0:
            print(prompt)
            print(model_answer)
            print(row[1]['label'])
    assert len(weak_labels) == len(validation_df)
    new_validation_df = validation_df.copy()
    new_validation_df['weak_label'] = weak_labels
    return new_validation_df


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['ctx'])):
        endings_choices = ast.literal_eval(example['endings'][i])
        prompt = f"### Instruction: Choose an appropriate ending for the provided sentence based on your commonsense knowledge.\
In your response, choose an answer from A,B,C,D, and provide a brief explanation. \
Sentence: {example['ctx'][i]}. \
Choices: A. {endings_choices[0]} B. {endings_choices[1]} C. {endings_choices[2]} D. {endings_choices[3]}. \
### Answer: {example['weak_label'][i]} "+ strong_tokenizer.eos_token
        output_texts.append(prompt)
    print(output_texts[:10])
    return output_texts



if __name__ == "__main__":

    args = parser.parse_args()

    weak_model_name = args.weak_model_name
    strong_model_name = args.strong_model_name

    df_val_w2s_path = "hellaswag_validation_w2s_naive_from_{}.csv".format(weak_model_name)

    if os.path.exists(df_val_w2s_path):
        df_val_new = pd.read_csv(df_val_w2s_path)
    else: 
        # load dataset
        df_val = pd.read_csv('dataset/validation.csv')
        # load weak model
        if weak_model_name=="Mistral-7B":
            base_model_path = "../models/mistralai_Mistral-7B-Instruct-v0.2"
            peft_model_path = "models/Mistral-7B-chat-hellaswag-train-sft"
            weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
        elif weak_model_name=="LLAMA2-7B":
            base_model_path = "../models/Llama-2-7b-chat-hf"
            peft_model_path = "models/Llama-2-7b-chat-hellaswag-train-sft"
            weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
        else:
            raise NotImplementedError
        # prepare dataset
        df_val_new = get_labels_from_weak_model(df_val, weak_model, weak_tokenizer)
        df_val_new.to_csv(df_val_w2s_path, index=False)
        # free up memory
        del weak_model, weak_tokenizer  

    df_sft_train, df_sft_val = np.split(df_val_new, [int(.8*len(df_val_new))])
    print("# of training samples: ", len(df_sft_train))
    print("# of validation samples: ", len(df_sft_val))
    print(df_val_new.head())
 
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
    print("Loaded base model from: ", base_model_path)

    # data collator
    response_template = ". ### Answer:"
    response_template_ids = strong_tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=strong_tokenizer)

    # train model
    trainer = train_model(strong_model, df_sft_train, df_sft_val, formatting_prompts_func, collator)

    # save model
    trainer.save_model("models/{}-hellaswag-SFT-on-weaksupervison-from-{}".format(strong_model_name, weak_model_name))

    # evaluate model
    accuracy, report = evaluate_super_alignment(strong_model, strong_tokenizer, "Hellaswag")

    # write to file
    with open("results/{}-hellaswag-SFT-on-weaksupervison-from-{}.txt".format(strong_model_name, weak_model_name), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.write(report)
        f.close()