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
        max_memory={0:"22GB",1:"22GB",2:"22GB",3:"22GB"},
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
    for i in range(len(example['input'])):
        text = f"### Instruction: The following are multiple choice questions about {example['subject'][i]}. \
In your response, choose an answer from A,B,C,D, and then provide a brief explanation. \
### Question: {example['input'][i]}. A. {example['A'][i]} B. {example['B'][i]} C. {example['C'][i]} D. {example['D'][i]}. \
### Answer: {example['target'][i]}. {example[example['target'][i]][i]} " + strong_tokenizer.eos_token
        output_texts.append(text)
    print(output_texts[:10])
    return output_texts

if __name__ == "__main__":

    args = parser.parse_args()

    strong_model_name = args.strong_model_name

    # prepare dataset
    with open('dataset/mmlu.pkl', 'rb') as handle:
        dataset_dict = pickle.load(handle)

    df_list = []
    for key in dataset_dict.keys():
        validation_df = dataset_dict[key]['validation'].to_pandas()
        validation_df['subject'] = " ".join(key.split('_'))
        df_list.append(validation_df)
    df = pd.concat(df_list, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train, df_val = np.split(df, [int(.9*len(df))])
    print("# of training samples: ", len(df_train))
    print("# of validation samples: ", len(df_val))
    print(df.head())    

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
    trainer.save_model("models/{}-mmlu-SFT-on-groundtruth".format(strong_model_name))

    # evaluate
    accuracy, report = evaluate_super_alignment(strong_model, strong_tokenizer, "MMLU")

    with open("results/{}-mmlu-SFT-on-groundtruth.txt".format(strong_model_name), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.write(report)
        f.close()