import argparse
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm
import datasets
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from peft import LoraConfig, get_peft_model
from transformers import  AutoTokenizer,AutoModelForCausalLM, TrainingArguments, Trainer
from trl import DataCollatorForCompletionOnlyLM

import sys
sys.path.append('..')
from evaluation import evaluate_super_alignment
from prompt_GPT import prompt_gpt_with_backoff

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

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        batch_weights = inputs.pop("answer_score")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        weights = batch_weights.unsqueeze(1).expand(-1, shift_logits.shape[1])
        weights = weights.contiguous().view(-1)
        loss_fct = CrossEntropyLoss(reduction='none')
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * weights
        loss = torch.mean(loss)
        return (loss, outputs) if return_outputs else loss


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



def train_model(model, df_train, df_val, data_collator):

    def preprocess_function(example):
        return strong_tokenizer(example["input"],truncation=True,max_length=512)

    ds_train = datasets.Dataset.from_pandas(df_train)
    ds_val = datasets.Dataset.from_pandas(df_val)

    ds_tokenized_train = ds_train.map(preprocess_function, batched=True)
    ds_tokenized_val = ds_val.map(preprocess_function, batched=True)

    ds_tokenized_train = ds_tokenized_train.remove_columns(["input"])
    ds_tokenized_val = ds_tokenized_val.remove_columns(["input"])


    training_args = TrainingArguments(
        output_dir='models/tmp',
        per_device_train_batch_size=2,
        gradient_accumulation_steps=12,
        learning_rate=args.lr,
        logging_steps=100,
        remove_unused_columns=False,)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        # max_seq_length=512,
        train_dataset=ds_tokenized_train,
        eval_dataset=ds_tokenized_val,
        data_collator=data_collator,
    )

    trainer.train()

    return trainer


def prepare_validation_set_GSM8K(weak_model_name): 
    df_val = pd.read_csv('gsm8k_validation_with_entropy_from_{}.csv'.format(weak_model_name))

    def get_answer_score(answer_value_counts, temperature=1.0):
        total_count = sum(answer_value_counts.values())
        answers_prob = [(float(v)/total_count)/temperature for k,v in answer_value_counts.items()]
        exp_sum = sum([np.exp(x) for x in answers_prob])
        answers_ = {k:np.exp((v/total_count)/temperature)/exp_sum for k,v in answers.items()}
        assert round(sum(list(answers_.values())),5) == round(sum(torch.nn.Softmax(dim=0)(torch.tensor(answers_prob)).tolist()),5)
        return answers_

    df_val["answer_score"] = ""

    for i in tqdm(range(0,len(df_val),5)):
        batch = df_val.loc[i:i+4]
        if batch['model_pred'].isnull().any():
            continue
        questions = batch['original_question'].unique()
        assert len(questions)==1  # assert only one question in a batch
        answers = batch['model_pred'].value_counts().to_dict()
        answers_ = get_answer_score(answers,temperature=0.2)
        for i in batch.index:
            batch.at[i,'answer_score'] = answers_[batch.at[i,'model_pred']]

    df_val = df_val[df_val['answer_score']!=""]
    df_val["input"] = df_val['prompt']+" "+df_val['weak_label']
    df_val = df_val.drop(columns=['original_question','original_answer','label','prompt','weak_label','model_pred', 'entropy'])
    return df_val


if __name__ == "__main__":
    args = parser.parse_args()

    # load model
    if args.strong_model_name == "Mistral-7B":
        base_model_path = '../models/mistralai_Mistral-7B-Instruct-v0.2'
    elif args.strong_model_name == "LLAMA2-13B":
        base_model_path = "../models/Llama-2-13b-chat-hf"
    elif args.strong_model_name == "LLAMA3-8B":
        base_model_path = "../models/Llama-3-8B"

    print("Loading base model from: ", base_model_path)
    strong_model, strong_tokenizer = load_model(base_model_path)

    # load data
    df_val = prepare_validation_set_GSM8K(args.weak_model_name)
    print("Loaded validation set: ", df_val.head())
    
    # sample data
    if args.do_sampling:
        df_val_new = df_val.sample(n = int(len(df_val)/5)).reset_index(drop=True)
    else:
        df_val_new = df_val.sample(frac=1).reset_index(drop=True)  # shuffle
    df_sft_train, df_sft_val = np.split(df_val_new, [int(.8*len(df_val_new))])

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    strong_model = get_peft_model(strong_model, peft_config)

    print("# of training samples: ", len(df_sft_train))
    print("# of validation samples: ", len(df_sft_val))

    # data collator
    response_template = ". ### Answer:"
    response_template_ids = strong_tokenizer.encode(response_template, add_special_tokens=False)[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=strong_tokenizer)

    # train model
    trainer = train_model(strong_model, df_sft_train, df_sft_val, collator)

    # save model
    trainer.save_model("models/{}_reweighting_from_{}_sample={}".format(args.strong_model_name, args.weak_model_name,args.do_sampling))

    # evaluate model
    accuracy = evaluate_super_alignment(strong_model, strong_tokenizer, "GSM8K")

    # write to file
    with open("results/{}_reweighting_from_{}_sample={}.txt".format(args.strong_model_name, args.weak_model_name,args.do_sampling), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.close()