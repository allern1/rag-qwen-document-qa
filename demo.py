import json
base = "."
with open(base + "/data/test_question.json", "r") as f:
    jdata1 = json.load(f)
    #jdata = json.loads(f.read())
print(jdata1)
#print(jdata1)