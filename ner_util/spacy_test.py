
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

ps = ["Can I have [word] [word] [p]","Can I order [word] [word] [p]","I want to order [word] [word] [p]","I want to have [word]  [word] [p]"]
#ps =["Can I have [word] [p]"]
# Add match ID "HelloWorld" with no callback and one pattern
#pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]
#pattern = [{"LOWER": "can"}, {"IS_SPACE": True}, {"LOWER": "i"}]
pattern = [{"LOWER": "can"}, {"LOWER": "we"}]
#pattern = [{"LOWER": "can"}, {"IS_SPACE": True}, {"LOWER": "I"}]

def token2pattern_element(token):
    mapper_d = {
        "[word]":{'IS_ALPHA': True},
        "[p]":{"IS_PUNCT": True}
    }
    if mapper_d.get(token):
        return mapper_d.get(token)
    else:
        return {"LOWER": token.lower()}

i = 0
for pattern_temp in ps:

    tokens = pattern_temp.split(" ")
    pattern_cur = []
    for token in tokens:
        pattern_cur.append(token2pattern_element(token))

    matcher.add("order"+str(i), None, pattern_cur)
    i+=1

#doc = nlp(u"can  i order some food! can we order    222")
doc = nlp(u"can I have some food. can i order some food,I want to have some fries")
matches = matcher(doc)
for match_id, start, end in matches:
    string_id = nlp.vocab.strings[match_id]  # Get string representation
    span = doc[start:end]  # The matched span
    print(match_id, string_id, start, end, span.text)
    print(span.text)