import json
import random

# load meta-data:
with open('val.jsonl', 'r') as file:
    reviews = file.read().split('\n')

for review in reviews[:-1]:
    # decode meta-data:
    data = json.loads(review)

    # load text:
    with open(f'docs/{data["annotation_id"]}', 'r') as file:
        txt = [sentence.split() for sentence in file.read().split('\n')]

    # select random sentence with evidence:
    evidences = data["evidences"][random.randint(0, len(data["evidences"])-1)]

    # select snippet including 3 sentences:
    sentences = list(range(
        max(evidences[0]["start_sentence"]-1, 0),
        evidences[0]["end_sentence"]+1
    ))

    # collect all evidences inside the snippet:
    for e in data["evidences"]:
        if (e[0]["start_sentence"] in sentences):
            if (e[0]["end_sentence"] in sentences):
                if e[0] not in evidences:
                    evidences.append(e[0])

    # combine tokens:
    tokens = []
    for i in sentences:
        try: tokens.extend(txt[i])
        except IndexError: pass

    # save sample:
    with open('val_sample.jsonl', 'a') as file:
        json.dump([{
            "annotation_id":data["annotation_id"],
            "text":' '.join(tokens), 
            "classification":data["classification"], 
            "evidences":evidences
        }], file)
        file.write('\n')