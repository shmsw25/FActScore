from factscore.factscorer import FactScorer

fs = FactScorer()

# this will create a database using your file
# for English Wikipedia (18GB)), it takes ~8 hours
# once DB file is created, you can reuse it by only specifying `db_path`
fs.register_knowledge_source("acl_corpus",
                             data_path="acl_corpus.jsonl",
                             db_path=None)

# # now, when you compute a score, specify knowledge source to use
# out = fs.get_score(topics, generations, knowledge_source=name_of_your_knowledge_source)
# print (out["score"]) # FActScore
# print (out["respond_ratio"]) # % of responding (not abstaining from answering)
# print (out["num_facts_per_response"]) # average number of atomic facts per response
