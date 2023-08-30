import pandas as pd
import tqdm
import json
import openai
from factscore.openai_lm import call_ChatGPT


df = pd.read_parquet('acl-publication-info.74k.parquet')
titles = df['title'].tolist()
abstracts = df['abstract'].tolist()
full_text = df['full_text'].tolist()
years = df['year'].tolist()
authors = [[y.strip() for y in x.split("and\n")] if x is not None else None for x in df['author'].tolist()]

# # build the corpus first
# output_corpus = []
# for title, abstract, ftext, author in zip(titles, abstracts, full_text, authors):
#     if author is not None and ftext.strip():
#         output_corpus.append({"title": title, "text": ftext})

# print(f"Number of papers in the corpus: {len(output_corpus)} ({len(titles) - len(output_corpus)} filtered)")

# # write the corpus to a jsonl file
# with open("acl_corpus.jsonl", 'w') as f:
#     for line in output_corpus:
#         f.write(json.dumps(line) + "\n")

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

# find all papers whose author is Kalpesh Krishna
for title, abstract, ftext, author in zip(titles, abstracts, full_text, authors):
    if title.strip() in prompt_titles:
        assert ftext.strip()

for title in prompt_titles:
    prompts_list.append(
        f"Give me a summary of the research paper titled \"{title}\"."
    )

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

# def count_freqs(counts, str_name="authors"):
#     counts = [(k, v) for k, v in counts.items()]
#     freq_list = [0, 1, 3, 10, 20, 50, 100000]
#     print("")
#     for i in range(len(freq_list) - 1):
#         num_counts = [x for x in counts if x[1] > freq_list[i] and x[1] <= freq_list[i + 1]]
#         print(f"Number of {str_name} with {freq_list[i]} < freq <= {freq_list[i + 1]}: {len(num_counts)} ({len(num_counts) / len(counts) * 100:.2f}%)")
#     print("")

# count_freqs(Counter(authors), "authors")

# all_entities = []
# for idx, abstract in tqdm.tqdm(enumerate(abstracts)):
#     doc = nlp(abstract)
#     curr_ents = []
#     for ent in doc.ents:
#         curr_ents.append(ent.text.strip())
#     curr_ents = list(set(curr_ents))
#     all_entities.append(curr_ents)

#     if (idx + 1) % 3000 == 0:
#         count_freqs(Counter([y for x in all_entities for y in x]), "entities")

# # write all_entities to a pickle file
# with open("indexes/acl_entities.pkl", 'wb') as f:
#     pickle.dump(all_entities, f)

# acl_counts = Counter([y for x in all_entities for y in x])
# # sort by frequency
# acl_counts = [(k, v) for k, v in acl_counts.items()]
# acl_counts = sorted(acl_counts, key=lambda x: x[1], reverse=True)

