import argparse
import string
import json
import numpy as np
import os

from tqdm import tqdm
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key

        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None

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
                  atomic_facts=None,
                  knowledge_source=None,
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
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.cache_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))

            if verbose:
                topics = tqdm(topics)

            new_topics, new_generations, new_atomic_facts = [], [], []
            for topic, gen in zip(topics, generations):
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    # abstain from answering
                    continue
                new_topics.append(topic)
                new_generations.append(gen)
                new_atomic_facts.append(curr_afs)
            topics, generations, atomic_facts = new_topics, new_generations, new_atomic_facts

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
            return np.mean(scores), atomic_facts

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
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()

    fs = FactScorer(args.model_name, args.cache_dir, args.openai_key)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.data_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break

    score = fs.get_score(topics=topics, generations=generations, atomic_facts=atomic_facts if args.use_atomic_facts else None, verbose=args.verbose)
    print (score)
    fs.save_cache()



