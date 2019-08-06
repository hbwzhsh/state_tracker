from transitions import Machine
import random
import itertools
from loggers import dbot_online_logger
from dialogue_state.domain import is_intent_rule
import json
from path_config import  *
## non_branched fsa
class buy_tv_state(object):
    pass

intents = ["buy_tv","inform_size","inform_resolution","inform_brandname"]
states=["start",'how can i help you' ]+[str(n+1)+"after_"+u for n,u in enumerate(intents)]
#print(states)

def get_states_from_transitions(transtions):
    states = []
    for t in transtions:
        states.append(t['source'])
        states.append(t['dest'])
    return list(set(states))

def load_edge2rule_by_taskname(taskname):
    edgename2condition = None
    with open(fsm_edge_rule_config.format(taskname)) as json_file:
        edgename2condition = json.load(json_file)
    return edgename2condition

def load_state2action_by_taskname(taskname):
    with open(fsm_state_action_config.format(taskname)) as json_file:
        state_action_mapper = json.load(json_file)
    return state_action_mapper

def load_transition_by_taskname(taskname):
    with open(fsm_transition_config.format(taskname)) as json_file:
        transition = json.load(json_file).get("transitions")
    return transition

def load_fsm_config_by_taskname(taskname):
    transition = load_transition_by_taskname(taskname)
    edgename2condition = load_edge2rule_by_taskname(taskname)
    state_action_mapper = load_state2action_by_taskname(taskname)
    states = get_states_from_transitions(transition)
    return transition,states,edgename2condition,state_action_mapper

tv_transition = [
    { 'trigger': 'buy_tv', 'source': 'start', 'dest': '1after_buy_tv' },
    { 'trigger': 'inform_size', 'source': '1after_buy_tv', 'dest': '2after_inform_size' },
    { 'trigger': 'inform_resolution', 'source': '2after_inform_size', 'dest': '3after_inform_resolution' },
    { 'trigger': 'inform_brandname', 'source': '3after_inform_resolution', 'dest': '4after_inform_brandname' },
    { 'trigger': 'greet', 'source': 'start', 'dest': 'how can i help you' },
    { 'trigger': 'buy_tv2', 'source': 'how can i help you' , 'dest': '1after_buy_tv' },
]
for i,state in enumerate(states):
    if state == "3after_inform_resolution" or state == "4after_inform_brandname":
        continue
    d = { 'trigger': "all_filled_"+str(i), 'source': state, 'dest': '4after_inform_brandname' }
    tv_transition.append(d)



def ask_next_slot_task_fsm(slots,start_state='start',end_state = 'end',slot_unfilled_reask = True):
    '''
    ask next unfilled slot
    :param slots:
    :param start_state:
    :param end_state:
    :param slot_unfilled_reask:
    :return:
    '''
    transition = []
    states = []
    n = len(slots)
    for i in range(n):
        d = {'trigger': 'any unfilled', 'source': "ask_next_one" + str(i - 1), 'dest': "ask_next_one" + str(i)}
        if i == 0:
            d = {'trigger': 'any unfilled', 'source': start_state , 'dest': "ask_next_one" + str(i)}

        transition.append(d)
        states.append(d['source'])
        states.append(d['dest'])
    for s in states[:]:
        d = {'trigger': 'all filled', 'source': s, 'dest': end_state}
        transition.append(d)
    states.append(end_state)
    states = list(set(states))
    states.sort()
    return transition,states

def ask_next_buy_tv():
    slots =["size", "brandname", "resolution"]
    transition,states = ask_next_slot_task_fsm(["size", "brandname", "resolution"])
    edge2condition = {
        'any unfilled':['any unfilled'],
        'all filled':[slot + " filled" for slot in slots],
    }
    state_action_map = {
        'ask_next_one0': ['action_ask_next_slot'],
        'ask_next_one1': ['action_ask_next_slot'],
        'ask_next_one2': ['action_ask_next_slot'],
        'ask_next_one3': ['action_ask_next_slot'],
        'end': ["utter_tv_summary","utter_send_tv_order"]}
    return states, None, transition, edge2condition, state_action_map


def n_edges_with_affirm_task_fsm(slots):
    filled = "filled"
    filled = "已填充"
    unfilled = "unfilled"
    unfilled = "未填充"
    chinese_affirm = "向用户确认槽位{}的值"
    english_affirm = "ask user whether the slot {} is xxx"
    transition = []
    states = []
    n = len(slots)
    for slot_index,slot in enumerate(slots):
        condition = slot + " "+filled
        if slot_index == n-1:
            dest_state = "finish"
            continue
        else:
            next_slot= slots[slot_index+1]

            dest_state ="ask_" + next_slot
        d = {'trigger': condition+"{} unfilled".format(next_slot), 'source': "ask_" + slot, 'dest': dest_state}
        d2 = {'trigger': slot + " "+filled, 'source': "ask_" + slot, 'dest': english_affirm.format(slot)}
        transition.append(d)
        transition.append(d2)
        states.append(d['source'])
        states.append(d2['source'])
        states.append(d['dest'])
        states.append(d2['dest'])

    states = list(set(states))
    states.sort()
    return transition, states

def n_edges_task_fsm(slots,start_state='start',end_state = 'end',slot_unfilled_reask = True):
    '''
    n edges design
    in one turn,the agent can move more than one node
    :param slots:
    :param start_state:
    :param end_state:
    :param slot_unfilled_reask:if a slot is unfilled,ask again
    :return:
    '''
    transition = []
    states = []
    n = len(slots)
    # slot a to slot b
    for slot_index,slot in enumerate(slots):
        condition = slot + " filled"
        if slot_index == n-1:
            dest_state = "finish"
            continue
        else:
            next_slot= slots[slot_index+1]

            dest_state ="ask_" + next_slot
        d = {'trigger': condition, 'source': "ask_" + slot, 'dest': dest_state}
        transition.append(d)
        states.append(d['source'])
        states.append(d['dest'])
    # check slot is filled or not
    for slot_index, slot in enumerate(slots):
        condition = slot + " unfilled"
        d = {'trigger': condition, 'source': "ask_" + slot, 'dest': "ask_" + slot}
        transition.append(d)
        states.append(d['source'])
        states.append(d['dest'])
    d = {'trigger': "any unfilled or intent is buy tv", 'source': start_state, 'dest': "ask_" + slots[0]}
    states.append(d['source'])
    states.append(d['dest'])
    transition.append(d)
    d = {'trigger': "all filled", 'source': "ask_" + slots[-1], 'dest':end_state }
    states.append(d['source'])
    states.append(d['dest'])

    transition.append(d)
    states = list(set(states))
    states.sort()
    return transition, states

def unfilled_fsm_policy(slots,start_state='start',end_state = 'end',add_start2nodes = True,add_nodes2end = True):
    '''
    n *(n-1) edges design
    :param slots:
    :param start_state:
    :param end_state:
    :param add_start2nodes: unfill one slot then ask
    :param add_nodes2end:  if fill all , then finish
    :return:
    '''
    transition = []
    states = []
    n = len(slots)
    perms = list(itertools.permutations([int(i) for i in range(n)], 2))
    for perm in perms:
        end_i = perm[1]
        miss_slotname = slots[end_i]
        last_miss_slotname = slots[perm[0]]
        condition = miss_slotname + " unfilled"
        d = {'trigger':condition , 'source': "ask_"+last_miss_slotname, 'dest': "ask_"+miss_slotname}
        transition.append(d)
        states.append(d['source'])
        states.append(d['dest'])
    states = list(set(states))
    if add_nodes2end:
        for source_node in states:
            d = {'trigger': "all filled", 'source': source_node, 'dest': end_state}
            transition.append(d)
    if add_start2nodes:
        for end_node in states:
            slot = end_node.split("_")[-1]
            d = {'trigger': slot+" unfilled", 'source': start_state, 'dest': end_node}
            transition.append(d)
    states += [start_state,end_state]
    return transition,states


def edge_rule_maker(slots):
    '''
    unfilled form policy ,
    make the rules for every possible edge
    :param slots:
    :return: dict
    '''
    edge2rules_dict = {}
    for slot in slots:
        edge2rules_dict[slot+" unfilled"] = [slot+" unfilled"]
    edge2rules_dict['all filled'] = [slot + " filled" for slot in slots]
    return edge2rules_dict

def n_edges_rule_maker(slots):
    '''
    unfilled form policy ,
    make the rules for every possible edge
    :param slots:
    :return: dict
    '''
    edge2rules_dict = {}
    for slot in slots:
        edge2rules_dict[slot+" filled"] = [slot+" filled"]
    edge2rules_dict["any unfilled"] = ["any unfilled"]
    edge2rules_dict['all filled'] = [slot + " filled" for slot in slots]
    return edge2rules_dict

def merge_dict(dicts):
    '''
    merge multiple dicts into one dict
    :param dicts: list[dict]
    :return: dict
    '''
    new = dict()
    for d in dicts:
        for k,v in d.items():
            new[k] = v
    return new



def simulate_form_policy(slots):
    transition = []
    states = []
    n = len(slots)
    string = "".join([str(i) for i in range(n)])
    a = list(itertools.permutations(string, n))
    #print(a)
    ##[('0', '1', '2'), ('0', '2', '1'), ('1', '0', '2'), ('1', '2', '0'), ('2', '0', '1'), ('2', '1', '0')]
    for perm in a:
        for i,slot_index_str in enumerate(perm):
            slot_index = int(slot_index_str)
            cur_slot = slots[slot_index]
            source_state = "ask_"
            slot_i_s = list(perm[:i])
            slot_i_s = [int(j) for j in slot_i_s]
            slotnames = [slots[j] for j in slot_i_s]
            #source_state+="".join(list(perm[:i]))
            source_state+="_".join(slotnames)
            if source_state == "ask_":
                dest_state = source_state+ slots[slot_index]
            else:
                dest_state = source_state+"_"+ slots[slot_index]
            states.append(source_state)
            states.append(dest_state)
            d = {'trigger': cur_slot+" unfilled " , 'source': source_state, 'dest':dest_state }
            transition.append(d)
    states = list(set(states))+["end"]
    for perm in a:
        slot_i_s = list(perm[:])
        slot_i_s = [int(j) for j in slot_i_s]
        slot_i_s = [slots[j] for j in slot_i_s]
        slot_i_s_str = "_".join(slot_i_s)
        for s in states:
            if slot_i_s_str in s:
                d = {'trigger': " all filled ", 'source': s, 'dest': "end"}
                transition.append(d)
    for state in states:
        if state!="end":
            d = {'trigger': " all filled ", 'source': state, 'dest': "end"}
            if d not in transition:
                transition.append(d)
    return transition,states



def reminder_data():
    transition ,states = unfilled_fsm_policy(["date","time","event"],end_state="end_add_new_event")
    d = {'trigger': 'intent is ask_reminder', 'source': 'start', 'dest': 'end_search_event'}
    transition.append(d)
    states += ['end_search_event']
    edgename2condition = edge_rule_maker(["date","time","event"])
    edgename2condition['intent is ask_reminder'] = ['intent is ask_reminder']
    state_action_map = {
        'ask_date': ["utter_ask_newevent_date"],
        'ask_time': ['utter_ask_newevent_time'],
        'ask_event': ["utter_ask_event"],
        'end_add_new_event': ["action_set_reminder"],
        'end_search_event':["action_report_event"]
    }
    return states, get_transition_data(transition), transition, edgename2condition, state_action_map




def n_edges_tv_data():
    form_policy, states = n_edges_task_fsm(["size", "brandname", "resolution"])
    edge2condition = n_edges_rule_maker(["size", "brandname", "resolution"])
    state_action_map = {
        'ask_brandname': ["utter_list_brand","utter_ask_brand"],
        'ask_resolution': ['utter_ask_resolution'],
        'ask_size': ["utter_list_size","utter_ask_size"],
        'end': ["utter_tv_summary","utter_send_tv_order"]}
    return states, get_transition_data(form_policy), form_policy, edge2condition, state_action_map

def new_tv_data():
    form_policy, states = unfilled_fsm_policy(["size", "brandname", "resolution"])
    edgename2condition = {'size unfilled': ['size unfilled'],
                          'brandname unfilled': ['brandname unfilled'],
                          'resolution unfilled': ['resolution unfilled'],
                          'all filled': ['size filled', 'brandname filled', 'resolution filled']}
    state_action_map = {
        'ask_brandname': ["utter_list_brand","utter_ask_brand"],
        'ask_resolution': ['utter_ask_resolution'],
        'ask_size': ["utter_list_size","utter_ask_size"],
        'end': ["utter_tv_summary","utter_send_tv_order"]}
    return states, get_transition_data(form_policy), form_policy, edgename2condition, state_action_map


def tv_data():
    form_policy, states = simulate_form_policy(["size", "brand", "resolution"])
    #print({k["trigger"]:[] for k in form_policy})
    edgename2condition = {'size unfilled ': ['size unfilled'],
                          'brand unfilled ': ['brandname unfilled'],
                          'resolution unfilled ': ['resolution unfilled'],
                          ' all filled ': ['size filled','brandname filled','resolution filled']}
    #print({k: [] for k in states})

    state_action_map = {
        'ask_size_brand_resolution': ['utter_ask_resolution'],
        'ask_brand': ["utter_list_brand","utter_ask_brand"],
        'ask_size_brand': ["utter_list_brand","utter_ask_brand"],
        'ask_resolution': ['utter_ask_resolution'],
        'ask_resolution_brand': ["utter_list_brand","utter_ask_brand"],
        'ask_resolution_brand_size': ["utter_list_size","utter_ask_size"],
        'ask_resolution_size': ["utter_list_size","utter_ask_size"],
        'ask_size_resolution_brand': ["utter_list_brand","utter_ask_brand"],
        'ask_brand_resolution': ['utter_ask_resolution'],
        'ask_resolution_size_brand': ["utter_list_brand","utter_ask_brand"],
        'ask_brand_size': ["utter_list_size","utter_ask_size"],
        'ask_size': ["utter_list_size","utter_ask_size"],
        'ask_size_resolution': ['utter_ask_resolution'],
        'ask_brand_resolution_size': ["utter_list_size","utter_ask_size"],
        'ask_brand_size_resolution': ['utter_ask_resolution'],
        'end': ["utter_tv_summary","utter_send_tv_order"]}

    return states,get_transition_data(form_policy) ,form_policy,edgename2condition,state_action_map

#tv_data()
#form_policy ,states = simulate_form_policy(["size","brand","resolution"])
condition2edge_name = dict()

## edge name --> rule set

edgename2condition = {
    "buy_tv":["size unfilled"],
    "greet":["intent is greet"],
    "buy_tv2":["size unfilled"],
    "inform_size":["resolution unfilled"],
    "inform_resolution":["brandname unfilled"],
    "inform_brandname":["size filled","resolution filled","brandname filled"],
}
state_action_map ={
    'how can i help you':["utter_offer_help"],
    "1after_buy_tv":["utter_list_size","utter_ask_size"],
    "2after_inform_size":["utter_receive_size","utter_ask_resolution"],
    "3after_inform_resolution":["utter_receive_resolution","utter_list_brand","utter_ask_brand"],
    "4after_inform_brandname":["utter_receive_brand","utter_tv_summary","utter_send_tv_order"],
}

def export_js_code(transition):
    datas = []
    for row in transition:
        start,end,edge = row["source"],row["dest"],row["trigger"]
        code = "g.setEdge(\""+ start+"\", \""+ end+"\", {label: \""+ edge+"\"});"
        #print(code)
        datas.append(code)
        #print(code)
    return (datas)

def get_transition_data(transition):
    datas = []
    for row in transition:
        start,end,edge = row["source"],row["dest"],row["trigger"]
        code = "g.setEdge(\""+ start+"\", \""+ end+"\", {label: \""+ edge+"\"});"
        data = [start,end,edge]
        datas.append(data)
        #print(code)
    return datas
# export_js_code(form_policy)
# print(states)

def cam_rest_data():
    '''
    开发人员读取预定义  fsm配置文件
    :return:
    '''

    transition = load_transition_by_taskname('camrest')
    states = get_states_from_transitions(transition)
    edgename2condition = load_edge2rule_by_taskname('camrest')
    state_action_map = load_state2action_by_taskname('camrest')
    return states, transition, edgename2condition, state_action_map


def weather_fsm_data():
    '''
    transition ,states,state-utters mapper,
    :return:
    '''
    states = []
    transition =load_transition_by_taskname('weather')

    states = get_states_from_transitions(transition)
    edgename2condition = load_edge2rule_by_taskname('weather')
    state_action_map = load_state2action_by_taskname('weather')

    return states,get_transition_data(transition),transition,edgename2condition,state_action_map
# print("----------------")
# weather_fsm_data()
# print("----------------")
class Fsm():
    def __init__(self,task):
        self.task = task
        states, data, transition, edgename2condition, state_action_map = 0,0,0,0,0
        if self.task == "buy_tv":
            states, data, transition, edgename2condition, state_action_map = ask_next_buy_tv()
            self.start_node = 'start'
        elif self.task == "camrest":
            states, transition, edgename2condition, state_action_map = cam_rest_data()
            self.start_node = 'state_ask_area'
        elif self.task == "weather":
        #states, data, transition, edgename2condition, state_action_map = weather_fsm_data()
            states, data, transition, edgename2condition, state_action_map = weather_fsm_data()
            self.start_node = 'start'
        elif self.task == "reminder":
            states, data, transition, edgename2condition, state_action_map = reminder_data()
            self.start_node = 'start'
        else:
            states, data, transition, edgename2condition, state_action_map = new_tv_data()
        # Initialize
        self.end_node = "4after_inform_brandname"

        self.buy_tv_task = buy_tv_state()
        self.transitions = transition
        self.states = states
        self.machine = Machine(self.buy_tv_task, states=self.states, transitions=self.transitions, initial=self.start_node)
        self.edgename2condition = edgename2condition
        self.state_action_map = state_action_map

    def return_start_node(self):
        self.set_current_state(self.start_node)

    def is_start_node(self):
        return self.current_state()==self.start_node

    def is_end_node(self,node):
        if node in ["1all filled","2all filled","end","end_search_event","end_add_new_event"]:
            return True
        return False

    def current_state(self):
        return self.buy_tv_task.state

    def set_current_state(self,state):
        self.buy_tv_task.state = state

    def is_current_state_end(self):
        return self.is_end_node(self.buy_tv_task.state)
    @staticmethod
    def get_edges_from_starting_state(transitions,start):
        ends = []
        for tran in transitions:
            if tran["source"] == start:
                ends.append(tran["trigger"])
        return  list(set(ends))


    def predict(self,edge_condition):
        if not edge_condition:
            dbot_online_logger.debug("there are not any edges")
            s = self.buy_tv_task.state
            return self.state_action_map.get(s)
        self.buy_tv_task.trigger(edge_condition)
        s = self.buy_tv_task.state
        dbot_online_logger.debug("current state is {}".format(s))
        return self.state_action_map.get(s)

    def get_matched_condition_edge(self,domain):
        if self.is_current_state_end():
            self.return_start_node()
        dest = []
        ends = Fsm.get_edges_from_starting_state(self.transitions,self.buy_tv_task.state)
        print("next step you can reach ",",".join(ends))
        for key in self.edgename2condition.keys():
            if key not in ends:
                #print(key , "not in dests ")
                continue
                #print(key, " in dests ")
            ruleset = self.edgename2condition[key]
            #print(ruleset)
            if domain.rule_set_evaluate(ruleset):
                dest.append(key)
        if len(dest)>1:
            for key in dest:
                ruleset = self.edgename2condition[key]
                for rule in ruleset:
                    if is_intent_rule(rule):
                        return key
            #print("your nlu condition matches {} edges".format(len(dest)))
            return dest[0]
        elif len(dest)==1:
            return dest[0]
        #print("your nlu condition does not match any edge with condition")
        return None


# F = Fsm()
# r = F.machine.get_transitions(source="start")
# print(r)
#
#
