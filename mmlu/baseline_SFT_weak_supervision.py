import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
        # torch_dtype=torch.float16,
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
        tokenizer=tokenizer,
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


def prompt_model(prompt, model, tokenizer, max_new_tokens=1):
    inputs =  tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
        outputs_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return outputs_string


def get_labels_from_weak_model(validation_df, model, tokenizer):
    weak_labels = []
    for row in tqdm(validation_df.iterrows(), total=len(validation_df)):
        prompt = f"### Instruction: The following are multiple choice questions about {row[1]['subject']}. \
In your response, choose an answer from A,B,C,D, and then provide a brief explanation. \
### Question: {row[1]['input']}. A. {row[1]['A']} B. {row[1]['B']} C. {row[1]['C']} D. {row[1]['D']}. \
### Answer:"
        model_answer = prompt_model(prompt, model, tokenizer, max_new_tokens=50)
        model_answer = model_answer.split("### Answer:")[1].strip()
        weak_labels.append(model_answer)
        if row[0]<=10:
            print(prompt)
            print(model_answer)
            print(row[1]['target'])
    assert len(weak_labels) == len(validation_df)
    new_validation_df = validation_df.copy()
    new_validation_df['weak_label'] = weak_labels
    return new_validation_df


# formatting prompts
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"### Instruction: The following are multiple choice questions about {example['subject'][i]}. \
In your response, choose an answer from A,B,C,D, and then provide a brief explanation. \
### Question: {example['input'][i]}. A. {example['A'][i]} B. {example['B'][i]} C. {example['C'][i]} D. {example['D'][i]}. \
### Answer: {example['weak_label'][i]} " +strong_tokenizer.eos_token
        output_texts.append(text)
    print(output_texts[:10])
    return output_texts



if __name__ == "__main__":

    args = parser.parse_args()

    weak_model_name = args.weak_model_name
    strong_model_name = args.strong_model_name

    # load weak model
    if weak_model_name=="Mistral-7B":
        base_model_path = "../models/mistralai_Mistral-7B-Instruct-v0.2"
        peft_model_path = "models/Mistral-7B-mmlu-train-sft"
        weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
    elif weak_model_name=="LLAMA2-7B":
        base_model_path = "../models/Llama-2-7b-chat-hf"
        peft_model_path = "models/Llama-2-7b-chat-mmlu-train-sft"
        weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
    else:
        raise NotImplementedError

    df_val_w2s_path = "MMLU_validation_w2s_naive_from_{}.csv".format(weak_model_name)
    if os.path.exists(df_val_w2s_path):
        new_validation_df = pd.read_csv(df_val_w2s_path)
    else: 
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
        new_validation_df = get_labels_from_weak_model(df, weak_model, weak_tokenizer)
        new_validation_df.to_csv(df_val_w2s_path, index=False)
    df_train, df_val = np.split(new_validation_df, [int(.9*len(new_validation_df))])
    print("# of training samples: ", len(df_train))
    print("# of validation samples: ", len(df_val))
    print(new_validation_df.head())    

    del weak_model, weak_tokenizer

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
    trainer = train_model(strong_model, strong_tokenizer, df_train, df_val, formatting_prompts_func, collator)

    # save model
    trainer.save_model("models/{}-mmlu-SFT-on-weaksupervision-from-{}".format(strong_model_name, weak_model_name))

    accuracy, report = evaluate_super_alignment(strong_model, strong_tokenizer, "MMLU")

    with open("results/{}-mmlu-SFT-on-weaksupervison-from-{}.txt".format(strong_model_name, weak_model_name), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.write(report)
        f.close()