# FActScore

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![PyPI version factscore](https://badge.fury.io/py/factscore.svg)](https://pypi.python.org/pypi/factscore/)
[![Downloads](https://pepy.tech/badge/factscore)](https://pepy.tech/project/factscore)

This is the official release accompanying our preprint, "FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation". FActScore is available as a PIP package as well.

## Install
<!-- ```
conda create -n fs-env python=3.9
conda activate fs-env
pip install -r requirements.txt
``` -->

Make a new Python 3.7+ environment using `virtualenv` or `conda`.

```bash
pip install factscore
python -m spacy download en_core_web_sm
```

## Download the data

```bash
python -m factscore.download_data
```

Or, download it manually from this [Google Drive link](https://drive.google.com/drive/folders/1bLHGu_imkZVtX6O0mpZ-G0-4ofTLM1ZA?usp=sharing). Make a cache directory `.cache/factscore`, and place unzipped `demos` and `enwiki-20230401.db` in that directory.

## Running FactScore

```bash
python -m factscore.factscorer --data_path {data_path} --model_name {estimator_name} --cache_dir {cache_dir} --openai_key {openai_key}
```

- `data_path` can be something like `data/unlabeled/InstructGPT.jsonl`. It should be a `.jsonl` format where each line contains `topic` (a topic entity that corresponds to the Wikipedia title) and `output` (a generation from the model).
- `model_name`: `retrieval+ChatGPT`, `retrieval+ChatGPT+npm`, two more configs (`retrieval+llama`, `retrieval+llama+npm`) coming soon!
- `cache_dir`: `.cache/factscore` by default.
- `openai_key`: File containing OpenAI API Key.
- `use_atomic_facts`: If specified, it uses model-generated atomic facts released as part of our data instead of running the atomic fact generator. You can't specify it if you are running new model generations.
- `n_samples`: If specified, it runs the model on a subset of the data.
- `verbose`: If specified, it shows the progress bar.

For example,

```python
python -m factscore.factscorer \
    --data_path data/unlabeled/InstructGPT.jsonl \
    --model_name "retrieval+ChatGPT" \
    --cache_dir ".cache/factscore" \
    --openai_key "api.key" \
    --verbose
```

It uses `enwiki-20230401` by default, and will download the database from our Google drive.

Instructions to use Instruct-LLAMA-7B or your own LM coming soon!

## To generate outputs from your own LM and evaluate them.

There're two sets of prompt entities, `data/labeled/prompt_entities.txt` (183 entities) and `data/unlabeled/prompt_entities.txt` (500 entities). Each line contains the name of the person (which is also a corresponding Wikipedia title). You can use the labeled version if you want to be compatible with the data under `data/labeled` (Section 3 and Section 4.2 in the paper), and use the unlabeled version if you want to be compatible with the data under `data/unlabeled` (Section 4.3 in the paper).

You can prompt your LM with your own prompt (we used `Question: Tell me a bio of <entity>.`) and create a .jsonl file, where each line has `topic` (entity name, exactly same as the one from `.txt` file) and `output` (generation from LM). This can be fed into `factscore.factscorer` using `--data_path`.


## To use a custom knowledge source.
You need a `.jsonl` file where each line is a dictionary containing `title` and `text`. `text` can either be a string or a list of strings (e.g., sections).

```python
from factscore.factscorer import FactScorer

fs = FactScorer()

# this will create a database using your file
# for English Wikipedia (18GB)), it takes ~8 hours
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source(name_of_your_knowledge_source,
                             data_path=path_to_jsonl_file,
                             db_path=path_to_output_db_file)

# now, when you compute a score, specify knowledge source to use
score = fs.get_score(topics, generations, knowledge_source=name_of_your_knowledge_source)
```






