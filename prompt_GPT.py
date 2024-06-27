import openai
import backoff
import time

openai.api_key = '' # put your openai api key here

def prompt_gpt(prompt,model_name="gpt-4o"): 
    try:
        completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content":prompt}
        ])
    except Exception as e:
        print(str(e))
        time.sleep(6)
        completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user", "content":prompt}
        ])
    return completion["choices"][0]["message"]["content"]


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def prompt_gpt_with_backoff(prompt,model_name="gpt-4o"):
    return prompt_gpt(prompt, model_name)