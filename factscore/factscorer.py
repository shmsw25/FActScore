import argparse
import string
import json
import numpy as np
import os
import subprocess

from tqdm import tqdm
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+llama+npm",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(cache_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.cache_dir, f"{name}.db")
        if data_path is None:
            data_path = os.path.join(self.cache_dir, f"{name}.jsonl")
        
        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")
        
        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))

    def get_score(self,
                  topics,
                  generations,
                  knowledge_source=None,
                  atomic_facts=None,
                  return_atomic_facts=False,
                  return_individual_decisions=False,
                  verbose=False):
        
        if knowledge_source is None:
            # use the default one (enwiki-20230401)
            knowledge_source = "enwiki-20230401"
            self.register_knowledge_source(knowledge_source)
        else:
            assert knowledge_source in self.retrieval, \
                f"{knowledge_source} is not registered yet. Please use `register_knowledge_source()` function to register it with a database"

        if type(topics)==len(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            raise NotImplementedError()

        if verbose:
            topics = tqdm(topics)

        scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            decision = self._get_score(topic, generation, facts, knowledge_source)
            score = np.mean([d["is_supported"] for d in decision])
            decisions.append(decision)
            scores.append(score)

        if return_atomic_facts:
            return np.mean(scores), [[d["atom"] for d in ds] for ds in decisions]

        if return_individual_decisions:
            return np.mean(scores), decisions

        return np.mean(scores)

    def _get_score(self, topic, generation, atomic_facts, knowledge_source):
        decisions = []
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                assert atom.endswith("."), atom
                prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())
                output = self.lm.generate(prompt)

                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom,
                              "is_supported": is_supported})

        return decisions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default="data/src-light/bio_PerplexityAI_v0.1.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    args = parser.parse_args()

    def extract_topic(dp):
        key_to_use = "prompt" if "prompt" in dp else "input"
        assert dp[key_to_use].startswith("Question: Tell me a bio of "), dp[key_to_use]
        topic = dp[key_to_use].split("\n")[0].split("Tell me a bio of")[-1].strip()
        assert topic.endswith(".")
        return topic[:-1]

    fs = FactScorer(model_name=args.model_name, cache_dir=args.cache_dir, openai_key=args.openai_key)
    af_generator = AtomicFactGenerator(key_path=args.openai_key, demon_dir=os.path.join(args.cache_dir, "demos"))

    topics = []
    generations = []
    atomic_facts = []

    tot = 0
    with open(args.data_path, "r") as f:
        for line in tqdm(f, desc="Getting atomic facts..."):
            tot += 1
            dp = json.loads(line)
            gen = ""
            facts = []
            if "response" in dp and dp["response"] is not None:
                # v0.1 files which have human written atomic facts
                response = dp["response"]
                for sent_idx, sent_dp in enumerate(response["fact_data"]):
                    gen += sent_dp["orig_sentence"] + " "
                    facts += sent_dp["orig_facts"]
                topics.append(extract_topic(dp))
                generations.append(gen)

            elif "output" in dp and dp["output"] is not None:
                # TODO: is processing para breaks critical for correct FactScore?
                atomic_facts, para_breaks = af_generator.run(dp["output"])
                for sent, afs in atomic_facts:
                    gen += sent + " "
                    facts.extend(afs)
                topics.append(extract_topic(dp))

            if facts:
                for i, fact in enumerate(facts):
                    # this should be manually fixed before releasing the data
                    if not fact.endswith("."):
                        facts[i] += "."

                atomic_facts.append(facts)

            if "ChatGPT" in args.model_name and tot == 100:
                break

    score, decisions = fs.get_score(topics,
                                    generations,
                                    atomic_facts=atomic_facts,
                                    return_individual_decisions=True,
                                    verbose=True)
    print ("%s\t%.1f" % (args.model_name, 100*score))

    fs.save_cache()

