import argparse
import os 
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

parser.add_argument(
    "--do_sampling",
    action="store_true",
    help="Whether to sample from the validation set or not. If True, sample the number of the original validation set.",
)

parser.add_argument(
    "--lr",
    default=0.00005,
    type=float,
    help="learning rate for training the model.",
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


def prompt_model(prompt, model, tokenizer, max_new_tokens=1):
    inputs =  tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=max_new_tokens)
        outputs_string = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]
    return outputs_string


def calculate_entropy(value_counts):
    total_count = sum(value_counts.values())
    pred_distribution = [pred/total_count for pred in value_counts.values()]
    if len(pred_distribution) < total_count:
        pred_distribution = pred_distribution + [0]*(total_count-len(pred_distribution))
    pred_entropy = entropy(pred_distribution)
    return pred_entropy


def generate_alternative_prompts(row, num_alternatives=5):
    alternative_prompts = []
    choices = ["0123","0132","0213","0231","0312","0321","1023","1032","1203","1230","1302","1320","2013","2031","2103","2130","2301","2310","3012","3021","3102","3120","3201","3210"]
    choices = np.random.choice(choices, num_alternatives, replace=False)
    endings_choices = ast.literal_eval(row[1]['endings'])
    label = row[1]['label']
    for choice in choices:
        A = endings_choices[int(choice[0])]
        B = endings_choices[int(choice[1])]
        C = endings_choices[int(choice[2])]
        D = endings_choices[int(choice[3])]
        target = choice.index(str(label))
        prompt =  f"### Instruction: Choose an appropriate ending for the provided sentence based on your commonsense knowledge.\
In your response, choose an answer from A,B,C,D, and provide a brief explanation. \
Sentence: {row[1]['ctx']}. \
Choices: A. {A} B. {B} C. {C} D. {D}. \
### Answer: "
        alternative_prompts.append((prompt, target, choice))
    return alternative_prompts


def get_df_val_with_entropy(df_val, weak_model, weak_tokenizer):
    new_df_list = []
    for row in tqdm(df_val.iterrows(), total=len(df_val)):
        alternative_prompts = generate_alternative_prompts(row)
        row_list = []
        ending_choices = ast.literal_eval(row[1]['endings'])
        for prompt, target,choice in alternative_prompts:
            answer = prompt_model(prompt, weak_model, weak_tokenizer, max_new_tokens=50)
            try:
                model_pred = answer.split("### Answer:")[1].strip(".-_ ")[0].upper()
            except:
                print("warning: model prediction is invalid")
                model_pred = ""
            if model_pred == 'A':
                model_pred = 0
            elif model_pred == 'B':
                model_pred = 1
            elif model_pred == 'C':
                model_pred = 2
            elif model_pred == 'D':
                model_pred = 3
            if isinstance(model_pred, int):
                model_pred_string = ending_choices[int(choice[model_pred])]
            else:
                model_pred_string = "None"
            if row[0]<10:
                print(prompt)
                print(answer.split("### Answer:")[1])
                print(model_pred)
                print(target)
                print("\n\n")
            row_list.append(( row[1]['ind'], target, prompt, answer.split("### Answer:")[1],model_pred_string))
        model_pred_list = [row_[4] for row_ in row_list]
        value_counts = pd.Series(model_pred_list).value_counts().to_dict()
        example_entropy = calculate_entropy(value_counts)
        row_list = [(row_[0], row_[1], row_[2], row_[3], row_[4], example_entropy) for row_ in row_list]
        new_df_list.extend(row_list)
    df_val_with_entropy = pd.DataFrame(new_df_list, columns  = ['ind', 'target', 'prompt', 'model_output', 'model_pred', 'entropy'])
    return df_val_with_entropy


def train_model(model, df_train, df_val, formatting_prompts_func, data_collator):

    ds_train = datasets.Dataset.from_pandas(df_train)
    ds_val = datasets.Dataset.from_pandas(df_val)

    training_args = TrainingArguments(
        output_dir='models/tmp',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=24,
        learning_rate=args.lr,
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
    for i in range(len(example['prompt'])):
        text = example['prompt'][i]+" "+example['model_output'][i] +strong_tokenizer.eos_token
        output_texts.append(text)
    print("examples of the SFT data")
    print(output_texts[:10])
    return output_texts


if __name__ == "__main__":  

    args = parser.parse_args()

    weak_model_name = args.weak_model_name
    strong_model_name = args.strong_model_name

    df_val_with_entropy_path = "hellaswag_validation_with_entropy_from_{}.csv".format(weak_model_name)
    if os.path.exists(df_val_with_entropy_path):
        df_val_with_entropy = pd.read_csv(df_val_with_entropy_path)
    else:
        # load dataset
        df_val = pd.read_csv('dataset/validation.csv')

        # load weak model
        if weak_model_name == "Mistral-7B":
            base_model_path = "../models/mistralai_Mistral-7B-Instruct-v0.2"
            peft_model_path = "models/Mistral-7B-chat-hellaswag-train-sft"
            weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
        elif weak_model_name == "LLAMA2-7B":
            base_model_path = "../models/Llama-2-7b-chat-hf"
            peft_model_path = "models/Llama-2-7b-chat-hellaswag-train-sft"
            weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
        else:
            raise NotImplementedError
        weak_model.eval()
        df_val_with_entropy = get_df_val_with_entropy(df_val, weak_model, weak_tokenizer)
        df_val_with_entropy.to_csv("hellaswag_validation_with_entropy_from_{}.csv".format(weak_model_name), index=False)
        # release GPU memory
        del weak_model


    # SFT strong model with validation examples which have low entropy
    entropy_threshold = np.percentile(df_val_with_entropy['entropy'], 50)
    print(f"Entropy threshold: {entropy_threshold}")
    # sample data
    if args.do_sampling:
        original_len = len(df_val_with_entropy)/5
        df_val_with_entropy = df_val_with_entropy[df_val_with_entropy['entropy']<=entropy_threshold].sample(n=int(original_len), replace=True)
    else:
        df_val_with_entropy = df_val_with_entropy[df_val_with_entropy['entropy']<=entropy_threshold]
    print(f"Number of examples with entropy less than {entropy_threshold}: {len(df_val_with_entropy)}")

    
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
    else:
        raise NotImplementedError
    print("Loaded strong model from: ", base_model_path)

    # train model
    response_template = ". ### Answer:"
    response_template_ids = strong_tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=strong_tokenizer)

    df_sft_train, df_sft_val = np.split(df_val_with_entropy.sample(frac=1), [int(.8*len(df_val_with_entropy))])
    sft_trainer = train_model(strong_model, df_sft_train, df_sft_val, formatting_prompts_func, collator)

    # save model
    sft_trainer.save_model("models/{}-hellaswag-w2sft-uncertainty-filtering-from-{}-sampling={}".format(strong_model_name, weak_model_name, args.do_sampling))

    # evaluate model
    accuracy, report = evaluate_super_alignment(strong_model, strong_tokenizer, "Hellaswag")
    
    # write to file
    with open("results/{}-hellaswag-w2sft-uncertainty-filtering-from-{}-sampling={}.txt".format(strong_model_name, weak_model_name, args.do_sampling), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.write(report)
        f.close()