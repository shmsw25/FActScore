import pandas as pd
import tqdm
import json
import openai
from factscore.openai_lm import call_ChatGPT
from factscore.factscorer import FactScorer

# File downloaded from https://github.com/shauryr/ACL-anthology-corpus
# https://drive.google.com/file/d/1CFCzNGlTls0H-Zcaem4Hg_ETj4ebhcDO/view?usp=sharing
df = pd.read_parquet('acl-publication-info.74k.parquet')
titles = df['title'].tolist()
full_text = df['full_text'].tolist()

acl_corpus = []
for x, y in zip(titles, full_text):
    if x.strip() == "" or y.strip() == "":
        continue
    acl_corpus.append({"title": x, "text": y})

with open("acl_corpus.jsonl", 'w') as f:
    for line in acl_corpus:
        f.write(json.dumps(line) + "\n")

fs = FactScorer()
# this will create a database using your file
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source("acl_corpus",
                             data_path="acl_corpus.jsonl",
                             db_path=None)


prompt_titles = [
    "Dense Passage Retrieval for Open-Domain Question Answering",
    "AmbigQA: Answering Ambiguous Open-domain Questions",
    "MetaICL: Learning to Learn In Context",
    "Noisy Channel Language Model Prompting for Few-Shot Text Classification",
    "Joint Passage Ranking for Diverse Multi-Answer Retrieval",
    "Reformulating Unsupervised Style Transfer as Paraphrase Generation",
    "Syntactically Supervised Transformers for Faster Neural Machine Translation",
    "Hurdles to Progress in Long-form Question Answering",
    "Generating Question-Answer Hierarchies",
    "Do Long-Range Language Models Actually Use Long-Range Context?"
]

prompts_list = []

for title in prompt_titles:
    prompts_list.append(f"Give me a summary of the research paper titled \"{title}\".")

with open("api.key", 'r') as f:
    api_key = f.readline()
openai.api_key = api_key.strip()

responses = []
for ptitle, prompt in tqdm.tqdm(zip(prompt_titles, prompts_list)):
    message = [{"role": "user", "content": prompt}]
    response = call_ChatGPT(message, model_name="gpt-3.5-turbo-0301")
    responses.append({
        "topic": ptitle,
        "output": response["choices"][0]["message"]["content"]
    })

# # write the corpus to a jsonl file
with open("acl_chatgpt_outputs.jsonl", 'w') as f:
    for line in responses:
        f.write(json.dumps(line) + "\n")
