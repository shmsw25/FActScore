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

- `data_path` can be something like `data/src-light/bio_ChatGPT_v0.2.jsonl` which is in a format we have been using so far. TODO for simplying the format and allowing it to take any topics/generations.
- `model_name`: `retrieval+llama`, `retrieval+llama+npm`, `retrieval+ChatGPT`, `retrieval+ChatGPT+npm`
- `cache_dir`: `.cache/factscore` by default.
- `openai_key`: File containing API Key, only needed when ChatGPT is being used.

For example,

```python
python -m factscore.factscorer \
    --data_path original_generation/v0.2/answers_mpt-7b_bio_test_addtional.jsonl \
    --model_name "retrieval+ChatGPT" \
    --cache_dir ".cache/factscore" \
    --openai_key "api.key"
```

It uses `enwiki-20230401` by default, and will download the database from our Google drive.
It also uses Inst-LLAMA, downloading from the Google Drive. TODO: need to release diff from LLAMA 7B only. Also need to allow users to specify their own LM path if they want to use a different LM.

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






