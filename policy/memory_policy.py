import numpy as np
from path_config import *
def sample_story():
    return {"name":"story1","turns":[
        {"user":"intent_hi","bot":"action_hi"},{"user":"intent_goodbye","bot":"action_goodbye"},{"user":"intent_goodbye","bot":"action_goodbye"}

    ]

    }

max_history = 1

def featurer(label,label_list):
    vec = [0.0]*len(label_list)
    if label==None or label not in label_list:
        if label:
            print(label+" no inside")
            print(label_list)
        return vec
    else:
        index  = label_list.index(label)
        vec[index] = 1.0
        return vec
def intent_his_to_vec(labels,domain_label):
    res = []
    for l in labels:
        vec = featurer(l,domain_label)
        res += vec
    return res

def intent_his_to_fix_turn_vec(labels,domain_label,n):
    res = intent_his_to_vec(labels, domain_label)
    normal_length = len(domain_label)* n
    if len(res)>normal_length:
        res = res[-normal_length:]
        return res
    padding = [0.0] * len(domain_label) * max(0, n - len(res)//len(domain_label))
    vec = padding + res
    return vec

def story2vector(s,max_history,intent_list):
    turns = s.get("turns")
    if turns:
        story_len = len(turns)
        states = []
        for slice_end,turn in enumerate(turns):
            slice_start = max(0, slice_end+1 - max_history)
            entire_his = turns[slice_start:slice_end]+[turns[slice_end]]
            entire_his_intent = [u["user"] for u in entire_his]
            vec = []
            for intent in entire_his_intent:
                cur = featurer(intent,intent_list)
                vec+=cur
            padding = [0.0]*len(intent_list) * max(0, max_history - (slice_end+1))
            vec = padding + vec
            states.append([vec,turn["bot"]])
        return states
    return None

def turns2states(turns,max_history):
    states = []
    for slice_end, turn in enumerate(turns):
        slice_start = max(0, slice_end + 1 - max_history)
        entire_his = turns[slice_start:slice_end] + [turns[slice_end]]
        entire_his_intent = [u["user"] for u in entire_his]
        vec = []
        for intent in entire_his_intent:
            vec.append(intent)
        padding = [None] * max(0, max_history - (slice_end + 1))
        vec = padding + vec
        states.append([vec, turn["bot"]])
    return states

def story2state(s,max_history,intent_list):
    turns = s.get("turns")
    if turns:
        states = turns2states(turns,max_history)
        return states
    return None

def get_memory_dict_from_storys(storys,max_history,intent_list):
    all_states = []
    for s in storys:
        #states = story2vector(s,max_history,intent_list)
        states = story2state(s, max_history, intent_list)
        all_states.extend(states)
    return all_states

def lookup(state_action_list,state):
    decoder = lambda num_list :"_".join([str(n) for n in num_list])
    d = {decoder(r[0]):r[1] for r in state_action_list}
    key = decoder(state)
    #print("look for {}".format(key))
    return d.get(key)

def predict( state_action_list , intent_his,n,domain_label = [] ):
    vec = intent_his_to_fix_turn_vec(intent_his,domain_label,n)
    for lin in state_action_list:
        #print(lin)
        array = np.array(lin[0]).reshape((n, len(domain_label)))
        for i in range(n):
            a = array[i].tolist()
            # if 1 in a:
            #     print(domain_label[a.index(1)])
            # else:
            #     print("no intent")

    return lookup(state_action_list,vec)

def get_rasa_story_path_by_task(task):
    story_path = any_story.format(task)
    return story_path

def rasa_md_loader_by_task(task):
    r = rasa_md_loader(get_rasa_story_path_by_task(task))
    return  r

def split_story(content):

    lines = content.splitlines(keepends=True)
    story = []
    storys = []
    for i,line in enumerate(lines):
        if "##" in line:
            if i != 0:
                storys.append(story[:])
            story = []
            #story.append(line)
        story.append(line)
        if i==len(lines) -1:
            storys.append(story[:])
    storys = ["".join(story) for story in storys]
    return storys

import path_config

def rasa_md_loader(path):
    story_ds = []
    intent_list = []
    action_list = []
    with open(path,mode="r") as w:
        content = w.read()
        #story_texts  = content.split("\n\n")
        story_texts  = split_story(content)
        # one story
        for s in story_texts:
            lines = s.splitlines()
            story_name = lines[0]
            user_intents = []
            bot_actions = []
            action_one_turn = []
            for i,l in enumerate(lines):
                # user intent
                if l.startswith("* "):
                    if i!=1:
                        bot_actions.append(action_one_turn[:])
                    action_one_turn = []
                    if "OR" not in l:
                        user_intents.append(l[2:].strip(" "))
                    else:
                        user_intents.append(l[2:].strip(" "))
                elif l.startswith(" - "):
                    action_one_turn.append(l[3:].strip(" "))
                    #bot_actions.append()
                if i == len(lines)-1:
                    bot_actions.append(action_one_turn[:])
            # assure len equal
            story_d = {}
            story_d["name"] = story_name[3:].strip(" ")
            story_d["turns"] = [{"user":q,"bot":a}for q,a in zip(user_intents,bot_actions)]
            intent_list.extend(user_intents)
            action_list.extend(bot_actions)
            story_ds.append(story_d)

    intent_list = list(set(intent_list))
    # action_list = list(set(action_list))
    return story_ds,intent_list,action_list

def read_intent_paths_from_file(path):
    paths = []
    story_ds = []
    intent_list = []
    action_list = []
    with open(path,mode="r") as w:
        content = w.read()
        #story_texts  = content.split("\n\n")
        story_texts  = split_story(content)
        # one story
        for s in story_texts:
            lines = s.splitlines()
            story_name = lines[0]
            user_intents = []
            bot_actions = []
            action_one_turn = []
            for i,l in enumerate(lines):
                # user intent
                if l.startswith("* "):
                    if i!=1:
                        bot_actions.append(action_one_turn[:])
                    action_one_turn = []
                    if "OR" not in l:
                        user_intents.append(l[2:].strip(" "))
                    else:
                        user_intents.append(l[2:].strip(" "))
                elif l.startswith(" - "):
                    action_one_turn.append(l[3:].strip(" "))
                    #bot_actions.append()
                if i == len(lines)-1:
                    bot_actions.append(action_one_turn[:])
            paths.append(user_intents)
    return paths
def build_tree_for_storydict(storydict,max_history):
    turns = storydict["turns"]
    root = Node()
    last_nodes  =[root]
    for t in turns:
        user = t["user"]
        bot = t["bot"]
        user_intents = [user]
        if "OR" in user:
            user_intents = user.split("OR")
        cur_nodes = []
        for intent in user_intents:
            intent = intent.replace(" ","")
            for last_node in last_nodes:
                cur = Node()
                cur.set_intent(intent)
                cur.set_parent(last_node)
                #cur.add_bot_action(bot)
                cur.bot_actions = bot
                cur_nodes.append(cur)
        last_nodes = cur_nodes[:]

    storys = []
    for node in last_nodes:
        nodes4path = node.get_nodes_along_path()
        nodes4path = [node for node in nodes4path if node.intent]
        #print(nodes_to_turns(nodes4path))
        storys.append(nodes_to_turns(nodes4path))
    all_states = []
    for turns in storys:
        states = turns2states(turns,max_history)
        all_states.extend(states)
    return last_nodes,all_states

def nodes_to_turns(nodes):
    return [{"user":n.get_intent(),"bot":n.get_bot_actions()}  for n in nodes]


def nodes_to_elements(nodes):
    d = dict()
    for node in nodes:
        k,v  = node.to_dict_element()
        d[k] = v
    return d

def nodes_to_story_paths(nodes):
    paths= []
    for node in nodes:
        k,v  = node.to_dict_element()
        paths.append([k,v])
    return paths

class Node():
    def __init__(self):
        self.parent = None
        self.children = []
        self.bot_actions = []
        self.intent = None

    def get_bot_actions(self):
        return self.bot_actions
    def set_parent(self,node):
        self.parent = node
        self.parent.add_child(self)

    def set_intent(self,intent):
        self.intent = intent

    def add_child(self,child):
        self.children.append(child)

    def node_list_to_keys(self):
        key_str = ""
        keys = [n.get_intent() for n in self.get_nodes_along_path()]
        keys  =[ k for k in keys if not k == None]

        return keys

    def to_dict_element(self):
        return self.node_list_to_keys(),self.bot_actions

    def add_bot_action(self,action):
        self.bot_actions.append(action)

    def get_intent(self):
        return self.intent

    def __str__(self):
        last = self
        res = ""
        #print("last intent",last.intent)
        if not last.intent:
            return "null node"
        while True:
            #print("last intent", last.intent)
            intent_name= last.intent
            if not intent_name:
                intent_name = "not intent"
            res +=(intent_name+" ")
            curent = last.parent

            if not curent:
                break
            last = curent

        return res

    def get_nodes_along_path(self):
        nodes = []
        last = self
        while True:
            nodes.append(last)
            curent = last.parent
            if not curent:
                break
            last = curent
        return reversed(nodes)

def build_state_dict(all_states):
    d = dict()
    for s in all_states:
        intents = s[0]
        intents = [intent for intent in intents if intent]
        intents_str = "|".join(intents)
        d[intents_str] = s[1]
    return  d

def get_all_intent(all_states):
    all_intents=[]
    for s in all_states:
        intents = s[0]
        intents = [intent for intent in intents if intent]
        all_intents.extend(intents)
    all_intents = list(set(all_intents))
    all_intents.sort()
    return all_intents

def predict_from_state_dict(d,state):
    intents_str = "|".join(state)
    return d.get(intents_str)

def get_all_intents_from_story_file(file_path):
    story_ds, _, _ = rasa_md_loader(file_path)
    all_states = []
    for story in story_ds:
        # print("story load one ")
        _, states = build_tree_for_storydict(story,1)
        all_states.extend(states)
    return  get_all_intent(all_states)

class MemoryPolicy():
    def __init__(self,path = "C://demo4rasa//data//stories2.md"):
        self.path = path
        self.max_history =1
        self.get_default_dict()

    def change_task(self,task):
        self.path = any_story.format(task)
        self.get_default_dict()

    def get_default_dict(self):
        story_ds,_,_ = rasa_md_loader(self.path)
        all_states = []
        for story in story_ds:
            #print("story load one ")
            _, states = build_tree_for_storydict(story,self.max_history)
            all_states.extend(states)
        self.states_dict = build_state_dict(all_states)
        self.all_intents = get_all_intent(all_states)
        #(self.states_dict)

    # intents contain no None object
    def predict_from_default_dict(self,intents):
        action = predict_from_state_dict(self.states_dict,intents[-self.max_history:])
        return action
# model_name = "buy_tv"
# story_path = "C://Users//Administrator//PycharmProjects//state_starker//template//stories_{}.md".format(model_name)
# r = MemoryPolicy(story_path).predict_from_default_dict(["greet"])
# print(r)
# paths = read_intent_paths_from_file(buy_tv_story)
# print(paths)