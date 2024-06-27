import argparse
import os 
import numpy as np
np.random.seed(42)
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments
import datasets
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from scipy.stats import entropy
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


def generate_alternative_prompts_GSM8K(row, num_alternatives=5):
    alternative_prompts = []
    GPT_prompt = f"The following is a grade-school level math question. \
Rewrite the question for {num_alternatives} times, and keep the semantics of the question unchanged. \
Directly return the results in {num_alternatives} lines, each line for one rewritten question. Question: {row[1]['question']}"
    GPT_answer = prompt_gpt_with_backoff(GPT_prompt, model_name="gpt-4o")
    GPT_answer = GPT_answer.split("\n")
    GPT_answer = [answer for answer in GPT_answer if answer.strip()] # remove empty strings from the GPT_answer
    GSM8K_prompt_template = "### Instruction: The following is a grade-school level math question. \
In your response, provide the numerical answer in the first line, and then provide a brief explanation in the second line. \
Question: {}. \
### Answer: "
    for i in range(num_alternatives):
        alternative_prompt = GSM8K_prompt_template.format(GPT_answer[i])
        alternative_prompts.append(alternative_prompt)
    return alternative_prompts



def get_df_val_with_entropy(df_val, weak_model, weak_tokenizer):
    new_df_list = []
    for row in tqdm(df_val.iterrows(), total=len(df_val)):
        alternative_prompts = generate_alternative_prompts_GSM8K(row)
        row_list = []
        for prompt in alternative_prompts:
            answer = prompt_model(prompt, weak_model, weak_tokenizer, max_new_tokens=50)  
            try:
                model_pred = answer.split("### Answer:")[1].strip().split("\n")[0].strip()
                if args.weak_model_name == "OPT-1.3B":
                    model_pred = model_pred.split(">>")[-1]
                model_pred = int(model_pred)
            except:
                model_pred = -100000000
            # if row[0]<10:
            print(prompt)
            print(answer.split("### Answer:")[1])
            print(model_pred)
            print("label:", row[1]["label"])
            print("\n\n")
            row_list.append((row[1]["question"],row[1]["answer"], row[1]["label"], prompt, answer.split("### Answer:")[1], model_pred))
        model_pred_list = [row_[5] for row_ in row_list]
        if -100000000 in model_pred_list:
            print("Warning: model prediction is not extracted correctly")
            continue
        value_counts = pd.Series(model_pred_list).value_counts().to_dict()
        example_entropy = calculate_entropy(value_counts)
        row_list = [(row_[0], row_[1], row_[2], row_[3], row_[4],row_[5], example_entropy) for row_ in row_list]
        new_df_list.extend(row_list)
    df_val_with_entropy = pd.DataFrame(new_df_list, columns  = ['original_question', 'original_answer', 'label', 'prompt', 'weak_label', 'model_pred' ,'entropy'])
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
        text = example['prompt'][i]+" "+example['weak_label'][i] +strong_tokenizer.eos_token
        output_texts.append(text)
    print("examples of the SFT data")
    print(output_texts[:10])
    return output_texts



if __name__ == "__main__":  

    args = parser.parse_args()

    weak_model_name = args.weak_model_name
    strong_model_name = args.strong_model_name

    df_val_with_entropy_path = "gsm8k_validation_with_entropy_from_{}.csv".format(weak_model_name)
    if os.path.exists(df_val_with_entropy_path):
        df_val_with_entropy = pd.read_csv(df_val_with_entropy_path)
    else:
        # load dataset
        df_val = pd.read_csv("dataset/validation.csv")

        # load weak model
        if weak_model_name == "Mistral-7B":
            base_model_path = "../models/mistralai_Mistral-7B-Instruct-v0.2"
            peft_model_path = "models/Mistral-7B-gsm8k-train-sft"
            weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
        elif weak_model_name == "LLAMA2-7B":
            base_model_path = "../models/Llama-2-7b-chat-hf"
            peft_model_path = "models/LLAMA2-7B-gsm8k-train-sft"
            weak_model, weak_tokenizer = load_model(base_model_path, peft_model_path)
        else:
            raise NotImplementedError

        weak_model.eval()
        df_val_with_entropy = get_df_val_with_entropy(df_val, weak_model, weak_tokenizer)
        df_val_with_entropy.to_csv(df_val_with_entropy_path, index=False)
        # release GPU memory
        del weak_model

    # SFT strong model with validation examples which have low entropy
    entropy_threshold = np.percentile(df_val_with_entropy['entropy'], 50)
    print(f"Entropy threshold: {entropy_threshold}")
    # sample data
    if args.do_sampling:
        original_len = len(df_val_with_entropy)/5
        df_val_with_entropy = df_val_with_entropy[df_val_with_entropy['entropy']<=entropy_threshold].sample(n=int(original_len))
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
    sft_trainer.save_model("models/{}-gsm8k-w2sft-uncertainty-filtering-from-{}-sampling={}".format(strong_model_name, weak_model_name, args.do_sampling))

    # evaluate model
    accuracy = evaluate_super_alignment(strong_model, strong_tokenizer, "GSM8K")
    
    # write to file
    with open("results/{}-gsm8k-w2sft-uncertainty-filtering-from-{}-sampling={}.txt".format(strong_model_name, weak_model_name, args.do_sampling), "w") as f:
        f.write("Accuracy: {}\n".format(accuracy))
        f.close()