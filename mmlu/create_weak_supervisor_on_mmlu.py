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
        max_memory={0:"10GB",1:"15GB",2:"24GB",3:"24GB"},
        torch_dtype=torch.bfloat16,
        )
    if peft_model_path:
        model.load_adapter(peft_model_path)
    return model, tokenizer


def train_model(model, tokenizer, df_train, df_val, formatting_prompts_func, data_collator):

    ds_train = datasets.Dataset.from_pandas(df_train)
    ds_val = datasets.Dataset.from_pandas(df_val)

    training_args = TrainingArguments(
        output_dir='models/tmp',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=24,
        learning_rate=0.00005,
        logging_steps=4,
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
        for i in range(len(example['input'])):
            text = f"### Instruction: The following are multiple choice questions about {example['subject'][i]}. \
### Question: {example['input'][i]}. A. {example['A'][i]} B. {example['B'][i]} C. {example['C'][i]} D. {example['D'][i]}. \
### Answer:{example['target'][i]} " +tokenizer.eos_token
            output_texts.append(text)
        print(output_texts[:10])
        return output_texts

    # prepare dataset
    with open('dataset/mmlu.pkl', 'rb') as handle:
        dataset_dict = pickle.load(handle)

    df_list = []
    for key in dataset_dict.keys():
        validation_df = dataset_dict[key]['train'].to_pandas()
        validation_df['subject'] = " ".join(key.split('_'))
        df_list.append(validation_df)
    df = pd.concat(df_list, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train, df_val = np.split(df, [int(.9*len(df))])
    print("# of training samples: ", len(df_train))
    print("# of validation samples: ", len(df_val))
    print(df.head())    

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
    trainer = train_model(model, tokenizer, df_train, df_val, formatting_prompts_func, collator)

    # save model
    trainer.save_model("models/{}-mmlu-train-sft".format(weak_model_name))
    
    # evaluate model
    accuracy, report = evaluate_super_alignment(model, tokenizer, "MMLU")

    with open("results/{}_weak_supervisor_performance.txt".format(weak_model_name), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.write(report)
        f.close()