from path_config import *
from policy.memory_policy import *
from loggers import dbot_online_logger
from model.slot_filling_model import *
from dialogue_state.entity import *

def unit_rule_reader(unit_rule_text):
    '''
    read a text form of unit rule and convert it into a normal rule object
    :param unit_rule_text:
    :return: rule object / rule dict
    '''
    #print(unit_rule_text)
    res = unit_rule_text.split(" ")
    if len(res) == 3:
        slotname , state,content  = res
    elif len(res) == 2:
        slotname, state = res
        content = "_"
    else :
        dbot_online_logger.debug("the rule is not valid : {}".format(unit_rule_text))
        slotname,state,content = None,None,None
    return slotname,state,content

def unit_intent_rule_reader(unit_rule_text):
    '''
    read a text form of unit rule and convert it into a normal rule object
    :param unit_rule_text:
    :return: rule object / rule dict
    '''
    res = unit_rule_text.split(" ")
    if len(res) == 3:
        intent,equal = res[-1],True
    elif len(res) == 4 and "not" in res:
        intent, equal = res[-1], False
    return intent,equal

def is_intent_rule(unit_rule_text):
    return "intent is" in unit_rule_text

small_intent_for_kvret = ["ask_how_weather", "inform_city", "ask_is_weather", "set_reminder", "ask_reminder", "inform_time",
                   "inform_date", "inform_event", "inform_partner", "ask_place", "ask_navigation", "thank",
                   "ask_address", "yes"]

weather = ["ask_how_weather", "inform_city", "ask_is_weather"]
reminder = ["set_reminder", "ask_reminder", "inform_time",
           "inform_date", "inform_event", "inform_partner"]
navigate = ["ask_place", "ask_navigation","ask_address"]
general = ["thank","yes"]
navigate_action = ['action_find_place', 'action_report_address', 'action_report_distance',
                   'action_set_navigation']  # ,
weather_action = ['action_check_weather']
schedule_action = ['action_report_event', 'action_set_reminder']
general_action = ['action_morehelp', 'action_goodbye', 'action_welcome']  # 'action_ok',
all_action = navigate_action + weather_action + schedule_action + general_action
def get_valid_user_act_from_intent(intent):
    if intent not in ["weather","schedule","navigate"]:
        raise ValueError("intent  should be in weather schedule or navigate")
    if intent=="weather":
        allow = weather
    elif intent =="schedule":
        allow = reminder
    elif intent == "navigate":
        allow = navigate
    allow += general
    return allow

def get_valid_action_from_intent(intent):
    if intent not in ["weather","schedule","navigate"]:
        raise ValueError("intent  should be in weather schedule or navigate")
    if intent=="weather":
        allow = weather_action
    elif intent =="schedule":
        allow = schedule_action
    elif intent == "navigate":
        allow = navigate_action
    allow += general
    return allow

def confirm_intent(intent):
    return "confirm_intent_"+intent

def confirm_slot(slotname):
    return "confirm_slot_"+slotname

def bot_confirm_intent(intent):
    return "bot_confirm_intent_"+intent

def bot_confirm_slot(slotname):
    return "bot_confirm_slot_"+slotname

def get_required_slot_names_by_task(taskname):
    '''
    read taskname from config json and
    then read the slot config file
     to get the slot names
    :param taskname: current task/story name
    :return: the slot names , list[str]
    '''
    filename = "task_{}.json".format(taskname)
    task_config_path = os.path.join(this_file_path, "template", filename)
    with open(task_config_path) as json_file:
        data = json.load(json_file)
    return data.get("required_slots")

#print(get_required_slot_names_by_task('buy_tv'))

def get_task_name_from_config():
    '''
    read json file config to get the task name
    :return:task name, str
    '''
    with open(story_config) as json_file:
        data = json.load(json_file)
    return data.get("task")

class Domain():
    def __init__(self):
        model_name = get_task_name_from_config()
        self.taskname = None
        # story_path = any_story.format(self.taskname)
        # self.required_slots = get_required_slot_names_by_task(self.taskname)
        #
        # self.intents = get_all_intents_from_story_file(story_path)

        self.do_spacy_ner = False
        self.nlu_model = SlotModel()
        self.nlu_model.load(rasa_nlu_model_path)
        self.task_ner_model = HelperNER()
        self.slots = None
        self.current_intent = None
        self.kb_data ={}
        self.user_q = ""

    def get_current_user_query(self):
        return self.user_q

    def set_current_user_query(self, q):
        self.user_q = q

    def set_kb_data(self,name,value):
        self.kb_data[name] = value

    def get_kb_data(self,name):
        return self.kb_data.get(name)

    def set_turn_intent(self,intent):
        self.current_intent = intent

    def get_turn_intent(self):
        return self.current_intent

    def get_required_slots(self):
        return get_required_slot_names_by_task(self.taskname)

    def get_lack_slotname_and_dict(self):
        required = self.get_required_slots()
        d = dict()
        for slot_name in required:
            d[slot_name] = self.find_slot_by_type( slot_name)
        lack_slot_names = [k for k, v in d.items() if v == None]
        print("------------------lack slots {}".format(",".join(lack_slot_names)))
        return lack_slot_names, d

    def update_slots_when_task_switch(self):
        self.required_slots = self.get_required_slots()

    def auto_intent_by_slot(self,slotnames):
        """
        if we get one slot called "xxx" from nlu and the current intent is unknown
        we can deduce that the intent is "inform_xxx"
        :param slotnames: all the slots we get
        :return: intent,str
        """
        for s in slotnames:
            if s.startswith("inform"):
                s = s[7:]
            for intent in self.intents:
                if s in intent:
                    return intent
        return None

    def is_finish_a_story_path(self,task,intents=None):
        if not intents:
            intents = self.intents
        paths = read_intent_paths_from_file(any_story.format(task))
        finish = False
        for path in paths:
            if self.is_match_story_path(intents,path):
                finish = True
        if not finish:
            dbot_online_logger.debug("intents and path")
            dbot_online_logger.debug("|".join(intents))
            for path in paths:
                dbot_online_logger.debug("path---"+"|".join(path))
        return finish


    def is_match_story_path(self,intents,path):
        if len(intents)  <  len(path):
            return False
        for intent_experience,intent_expected in zip(intents,path):
            if intent_experience not in intent_expected:
                return False
        dbot_online_logger.debug("intents and path")
        dbot_online_logger.debug("|".join(intents))
        dbot_online_logger.debug("path--- "+"|".join(path))
        return True

    def slot_affirm(self,parsing_result):
        entities = parsing_result['entities']
        es = []
        need_2_confirm = [e_dict for e_dict in entities if slot_min_threshold<e_dict["confidence"] < slot_threshold]
        domain_slots = self.get_required_slots()
        domain_slots += ["inform_"+u for u in domain_slots]
        need_2_confirm = [e_dict for e_dict in need_2_confirm if e_dict["entity"] in domain_slots]
        if need_2_confirm==[]:
            #not need to confirm
            return None,[]
        print("found slots uncertain")
        res = "the uncertain entities will shown below,please input the entity name to confirm:\n"
        res+=",".join(["{}:{}".format(e_dict["entity"],e_dict["value"]) for e_dict in need_2_confirm])
        return res,need_2_confirm

    def get_entity(self, sentence):
        es =HelperNER().get_entity(sentence)
        if self.taskname:
            domain_slots = self.get_required_slots()
            domain_slots += ["inform_"+u for u in domain_slots]
            return [e for e in es if e.get_entity_type() in domain_slots]
        else:
            return es

    def get_valid_slots_from_nlu_result(self,nlu_result,q,t):
        current_slots = self.nlu_model.rasa2slots(nlu_result, t)
        for s in current_slots:
            print(s)
        if self.do_spacy_ner:
            basic_entities = GeneralNer.get_basic_entity(q)
            current_slots = current_slots + basic_entities
        domain_slots = self.get_required_slots()
        domain_slots += ["inform_"+u for u in domain_slots]
        current_slots = [slot for slot in current_slots if slot.get_entity_type() in domain_slots]
        return current_slots

    def is_slots_filled(self, taskname,tracker):
        required_slot_names = get_required_slot_names_by_task(taskname)
        #required_slot_names = ["inform_"+u for u in required_slot_names]
        dbot_online_logger.debug("check task state:")
        have_filled_slots = set(tracker.get_filled_slot_names())
        dbot_online_logger.debug("|".join(list(have_filled_slots)))
        if set(required_slot_names).intersection(have_filled_slots) ==  set(required_slot_names):
            return True
        return False

    def set_slots(self,slots):
        new_slots = [s.copy() for s in slots]
        new_slots = [slot.cut_inform() for slot in new_slots]
        self.slots = new_slots


    def find_slot_by_type(self,slot_type):
        for slot in self.slots:
            if slot.get_entity_type() == slot_type:
                return slot
        return None

    def rule_evaluate(self,rule_text):
        if  is_intent_rule(rule_text):
            intent,equal_relation = unit_intent_rule_reader(rule_text)
            is_match = False
            if intent == self.get_turn_intent():
                is_match = True
            if equal_relation == False:
                is_match = not is_match
            return is_match

        ## not a intent rule


        slot_name ,state, content = unit_rule_reader(rule_text)
        slot = self.find_slot_by_type(slot_name)

        # if not slot:
        #     print("warning : there is not such slot {}".format(slot_name))
        valid_slot_states = ["filled", "unfilled", "contain", "not_contain", "equal", "not_equal"]
        if state not in valid_slot_states:
            print("warning : there is not such state {}".format(state))
            return True
        ## all input correct
        if state == "any unfilled":
            task = self.taskname
            slots = get_required_slot_names_by_task(task)
            return any([self.find_slot_by_type(s)==None  for s in slots])
        if state == "filled":
            return  slot!=None
        elif state == "unfilled":
            return not slot
        elif not slot:
            print("warning : there is not such slot {}".format(slot_name))
            return False
        elif not content:
            print("warning : there is not  content ")
            return False
        elif state == "contain":
            return content in slot.get_entity_value()
        elif state == "not_contain":
            return not  content in slot.get_entity_value()
        elif state == "equal":
            return content == slot.get_entity_value()
        elif state == "not_equal":
            return not content == slot.get_entity_value()
        else:
            print("no valid rules")
            return True

    def rule_set_evaluate(self,rule_set):
        res = True
        for rule in rule_set:
            cur = self.rule_evaluate(rule)
            # if cur:
            #     print(" match the rule ",rule)
            # else:
            #     print(" does not match the rule ", rule)
            res = (res and cur)
        return res







