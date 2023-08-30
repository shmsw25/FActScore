import csv
import json

# read json file acl_chatgpt_outputs_factscore_output.json

with open('acl_chatgpt_outputs_factscore_output.json') as f:
    data = json.load(f)

output_csv = []

for idx, instance in enumerate(data['decisions']):
    for atomic in instance:
        output_csv.append({
            'atom': atomic['atom'],
            'factscore_is_supported': atomic['is_supported'],
            'instance_id': idx
        })

# write to csv
with open('acl_chatgpt_outputs_factscore_output.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=['instance_id', 'atom', 'factscore_is_supported'])
    writer.writeheader()
    writer.writerows(output_csv)
