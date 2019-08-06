#from model.action_model import RnnBinaryModel,Glove
from action_api.schdule import ScheduleManager
# from Datebase import DatabaseManger
from dialogue_state.tracker_manager import TrackerManager
from dialogue_state.action import Action
from dialogue_state.domain import *
from dialogue_state.task_action import Task, Navigate


class Agent():
    def __init__(self,manager = None ,db_manager = None):
        """
        init and load model here，rasa model for slot filling ,keras basic dense multi label classifcatioin for multi-action
        """
        camb_rest_slot_path = 'C:\\Users\\Administrator\\PycharmProjects\\Dialogue_Bot\\models\\current\\CamRest_slot_filling'
        self.slot_model = SlotModel()
        self.slot_model.load(camb_rest_slot_path)
        self.action_model = RnnBinaryModel()
        self.action_model.load_models()
        self.db_manager  = db_manager
        #self.action_model.load_model(self.action_model.model_path)
        self.trackers_manager = manager

        self.enable_slot_affirm = True
        self.enable_intent_affirm = True
        self.normal_mode = True
        self.input_slot = False
        self.input_intent = False
        self.need2confrimslot = []
        self.print_info = []  # ["action","template"]

        self.domain = None

    def actions_update(self,current_tracker,action,updated_slots):
        """
        :param current_tracker: tracker for bot
        :param action: actions list
        :param updated_slots: slot obtained from the user q
        :return: reply text list
        """
        reply_list = []
        if isinstance(action, list):
            for action_name in action:
                #print("running ", action_name)
                # event.set_intent(intent)
                i = 0
                bot_action = Navigate(action_name, {}, self.reminder_manager,self.domain)
                bot_action.slots = updated_slots
                reply = bot_action.run()
                # reply = Navigate.get_tmp_by_action(action_name)
                if not reply:
                    reply = self.action_tmp(action_name)
                current_tracker.add_bot_event(reply, slotset=updated_slots,action_name=action_name)
                if not reply:
                    result = ""
                current_tracker.last_action = action_name
                reply_list.append(reply)
        dbot_online_logger.debug("reply is [{}]".format("|".join(reply_list)))
        return reply_list

    def user_input_intent_to_confirm(self, q):
        '''
         the agent is  typing the intent to affirm the intent detection in last turn
        :param q: user message ,contain the intent name
        :return: the reply from bot
        '''
        dbot_online_logger.debug("======= mode : typing/receiving the intent for confirm ======")
        intent = q
        if "intent" in self.print_info:
            print("updated intent --->[%s]" % intent)
        current_tracker = self.default_tracker()
        intent_so_far = current_tracker.get_intent_sofar(intent)
        #event = current_tracker.add_user_event(q, slots=current_tracker.slotset, intent=intent)
        event = current_tracker.add_user_confirm(q, slots=current_tracker.slotset, intent=confirm_intent(intent))
        nlu_result = current_tracker.get_last_nlu_result()
        dbot_online_logger.debug("intent_so_far")
        dbot_online_logger.debug("|".join(intent_so_far))
        action = self.predict_action(intent_so_far)
        if "action" in self.print_info:
            print("= action is = =", action)
        # intent for this turn ,slot for slot collectted from the begining
        reply_list = self.actions_update(current_tracker, action, current_tracker.slotset)
        replys = "\n".join(reply_list)
        self.input_intent = False
        print("finish confirm intent")
        return replys, "??"

    def user_input_slot_to_confirm(self, q):
        '''
         the agent is  typing the slot to affirm the slot filling in last turn
        :param q: user message ,contain the intent name
        :return: the reply from bot
        '''
        ds = self.need2confrimslot
        ds = [d for d in ds if d["entity"] in q]
        dbot_online_logger.debug("confrim the slots {}".format(",".join(["{}:{}".format(d["entity"], d["value"]) for d in ds])))
        #print("confrim the slots {}".format(",".join(["{}:{}".format(d["entity"], d["value"]) for d in ds])))
        current_slots = [Entity(d["entity"], d["value"]) for d in ds]
        current_slotnames = [d["entity"] for d in ds]
        current_tracker = self.default_tracker()
        # perfect
        inform_slots = [s.inform_slot() for s in current_slots]
        updated_slots = current_tracker.update(current_tracker.slotset, inform_slots)
        intent_so_far = current_tracker.get_past_intent()
        #event = current_tracker.add_user_event(q, slots=updated_slots, intent=current_tracker.get_last_intent())
        slots_str = "[{}]".format(",".join(current_slotnames))
        event = current_tracker.add_user_confirm(q, slots=current_tracker.slotset, intent=confirm_slot(slots_str))
        current_tracker.slotset = updated_slots
        action = self.predict_action(intent_so_far)
        dbot_online_logger.debug("actions are [{}]".format(action))
        if "action" in self.print_info:
            print("= action is = =", action)
        reply_list = self.actions_update(current_tracker, action, current_tracker.slotset)
        replys = "\n".join(reply_list)
        self.input_slot = False
        print("finish confirm slot")
        return replys, "??"

    def bot_do_slot_confirm(self, q, intent, nlu_result):
        """
        bot reply to ask slot confirm from user
        :param q:user query
        :param intent:
        :param nlu_result:rasa nlu parsing result or other equivalent format
        :return:bot reply ,and action name
        """
        affirm_reply, self.need2confrimslot = self.domain.slot_affirm(nlu_result)
        if affirm_reply:
            slots = [e_dict["entity"] for e_dict in self.need2confrimslot]
            slots.sort()
            self.input_slot = True
            current_tracker = self.default_tracker()
            #event = current_tracker.add_user_event(q, slots=current_tracker.slotset, intent=intent)
            event = current_tracker.add_bot_confirm(affirm_reply, action_name=bot_confirm_slot(",".join(slots)))
            print("add bot slot confirm")
            return affirm_reply, "??"
        return None,None

    def is_need_handle_user_affirm(self):
        '''
        whether we need to handle last turn affirm from bot
        :return: T.F
        '''
        return self.input_slot or self.input_intent

    def handle_user_affirm(self, q):
        reply_list_intent , reply_list_slot = [],[]
        # going to input the input
        current_tracker = self.default_tracker()
        res = ""
        intent_so_far = []
        if self.input_intent :
            dbot_online_logger.debug("======= mode : typing/receiving the intent for confirm ======")
            #intent = q
            intent = self.domain.nlu_model.get_domain_intent_from_q(q,self.domain.taskname)
            if not intent:
                dbot_online_logger.debug("no valid intent is found.")
                return "no valid intent",""
            if "intent" in self.print_info:
                print("updated intent --->[%s]" % intent)
            intent_so_far = current_tracker.get_intent_sofar(intent)
            # event = current_tracker.add_user_event(q, slots=current_tracker.slotset, intent=intent)
            event = current_tracker.add_user_confirm(q, slots=current_tracker.slotset, intent=confirm_intent(intent))
            nlu_result = current_tracker.get_last_nlu_result()
            dbot_online_logger.debug("intent_so_far")
            dbot_online_logger.debug("|".join(intent_so_far))
            # intent for this turn ,slot for slot collectted from the begining
            self.input_intent = False
            print("finish confirm intent")

        # going to input the slot
        if  self.input_slot:
            dbot_online_logger.debug("======= mode : typing/receiving the slot for confirm ======")
            ds = self.need2confrimslot
            ds = [d for d in ds if d["entity"] in q]
            print("confrim the slots {}".format(",".join(["{}:{}".format(d["entity"], d["value"]) for d in ds])))
            current_slots = [Entity(d["entity"], d["value"]) for d in ds]
            current_slotnames = [d["entity"] for d in ds]
            # perfect
            inform_slots = [s.inform_slot() for s in current_slots]
            updated_slots = current_tracker.update(current_tracker.slotset, inform_slots)
            intent_so_far = current_tracker.get_past_intent()
            #intent_so_far = current_tracker.get_intent_sofar(intent)
            # event = current_tracker.add_user_event(q, slots=updated_slots, intent=current_tracker.get_last_intent())
            slots_str = "[{}]".format(",".join(current_slotnames))
            event = current_tracker.add_user_confirm(q, slots=current_tracker.slotset, intent=confirm_slot(slots_str))
            current_tracker.slotset = updated_slots
            dbot_online_logger.debug("----slots----")
            for slot in current_tracker.slotset:
                dbot_online_logger.debug("slot:{}".format(slot.__str__()))
            self.input_slot = False
            print("finish confirm slot")

        action = self.predict_action(intent_so_far)
        if "action" in self.print_info:
            print("= action is = =", action)
        reply_list = self.actions_update(current_tracker, action, current_tracker.slotset)
        replys = "\n".join(reply_list)
        return replys,"??"



    def is_need_intent_affirm(self,nlu_result,q):
        have_done = self.default_tracker().have_done_user_intent_affirm()
        return self.enable_intent_affirm and (not have_done) and self.domain.nlu_model.is_have_uncertain_intent(self.domain.taskname,q)

    def is_need_slot_affirm(self,nlu_result):
        return self.enable_slot_affirm and nlu_result and nlu_result.get("entities")!=[]and self.domain.nlu_model.is_have_uncertain_slot(nlu_result)

    def do_bot_affirm(self,user_q,nlu_result):
        affirm_reply_for_intent,affirm_reply_for_slot ="",""
        intent =  self.domain.nlu_model.rasa_intent_with_domain(user_q, self.domain.taskname)
        action_name = ""
        if self.is_need_intent_affirm(nlu_result,user_q):
            dbot_online_logger.debug("is_need_intent_affirm")
            affirm_reply_for_intent,candidate_intents = self.domain.nlu_model.intent_affirm(self.domain.taskname,user_q)
            self.input_intent = True
            intent = "undefined"
            action_name += "bot_confirm_intent_"
        if self.is_need_slot_affirm(nlu_result):
            dbot_online_logger.debug("is_need_slot_affirm")
            self.input_slot = True
            affirm_reply_for_slot, self.need2confrimslot = self.domain.slot_affirm(nlu_result)
            # get affirm reply
            slots = [e_dict["entity"] for e_dict in self.need2confrimslot]
            slots.sort()
            action_name += "bot_confirm_slot_"
        if not affirm_reply_for_intent :
            affirm_reply_for_intent = ""
        if not affirm_reply_for_slot:
            affirm_reply_for_slot = ""
        res = affirm_reply_for_intent +"\n"+affirm_reply_for_slot
        self.default_tracker().add_user_event(user_q,parsing_result=nlu_result, slots=self.default_tracker().slotset, intent=intent)
        self.default_tracker().add_bot_confirm(res, action_name=action_name)
        return res,None

    def default_tracker(self):
        return self.trackers_manager.current_tracker()

    def action_tmp(self,action):
        return " I am doing action {}".format(action)

    def process(self,q):
        parsing_result = self.slot_model.predict(q)
        current_tracker = self.default_tracker()
        user_event = current_tracker.add_user_event(q, parsing_result)
        if self.db_manager:
            self.db_manager.insert_event_mongo(user_event.event2dict('test_bot'))

        slu = self.slot_model.rasa2slu(parsing_result)
        previous_slots = current_tracker.slotset
        current_slots = current_tracker.rasa_to_slots(parsing_result,0.9)
        print('========current_slots======')
        for s in current_slots:
            s.print_entity()
        slots = current_tracker.update(previous_slots ,current_slots)
        print('========updated slots======')
        for s in slots:
            s.print_entity()
        history = [e.get_slotset() for e in current_tracker.events]
        history.append(slots)
        #action_name = self.action_model.get_next_action_from_slots(slots)
        action_name = self.action_model.get_next_action_from_slot_history(history)
        i = 0
        if action_name == 'action_search_rest':
            print('========updated slots2======')
            for s in slots:
                s.print_entity()
            slot_names = [s.type_name for s in slots if s.type_name.startswith('inform_')]
            task = Task('action_search_rest')
            result, rests = task.run(slots, slot_names)
            if rests:
                for k in rests[0].keys():
                    slot = Entity('found_' + k)
                    slot.set_value(rests[0].get(k))
                    slots.append(slot)
            current_tracker.slotset = slots
        else:
            print('========updated slots2======')
            for s in slots:
                s.print_entity()
            action = Action(action_name,{})
            result = action.get_filled_response(current_tracker.slotset)
        return result+'({})'.format(action_name)

class Helper(Agent):

    def  __init__(self,manager = None ,db_manager = None,reminder_manager = None):
        """
        init and load model here，rasa model for slot filling ,keras basic dense multi label classifcatioin for multi-action
        """
        self.slot_model = HelperNER()
        self.action_model = Glove()
        #self.model_path = "C://model/pretrain_kvret_.h5"
        self.model_path = "C://model/pretrain_kvret3_.h5"

        self.action_model.all_action = self.action_model.kvret_actions()
        self.db_manager  = db_manager
        self.reminder_manager  = reminder_manager
        #self.action_model.load_model(self.action_model.model_path)
        self.trackers_manager = manager
        self.action_model.load_model(self.model_path)
        self.inform_slot = ['location','weather_attribute','date','time','date','event']
        self.intent = "weather"
        self.kvret_sf = kvret_sf_intent()
        self.kvret_sf.load(self.kvret_sf.model_path + "\\default\\" + self.kvret_sf.model_name)

    def set_intent(self,intent):
        self.intent = intent

    def get_force_action_with_query(self, q):
        # _actionxxx:query
        if ":" not in q:
            raise ValueError(" \":\" should be inside the query but query is {}".format(q))

        action = q.split(":")[0][1:]
        query = q.split(":")[1]
        return action,query

    def system_call(self,q):
        if q.startswith('intent:'):
            intent = q.split(":")[1]
            self.set_intent(intent)
            return 'update {}'.format(intent),"_"
        elif q == "#clear":
            self.default_tracker().clear_event()
            return "clear events","_"

    def process(self,q):
        force_act = None
        if q.startswith('_'):
            force_act,query = self.get_force_action_with_query(q)
        else:
            reply,a = self.system_call(q)
        if force_act:
            q = query
        current_slots = self.slot_model.get_entity(q)
        if len(q.split(" "))>3:
            nlu_result = self.kvret_sf.predict(q)
            intent = nlu_result["intent"]["name"]
            print("updated %s"%intent)
        else:
            intent  = None
            print("not updated %s" % self.intent)
        if intent:
            self.intent = intent
        current_tracker = self.default_tracker()
        previous_slots = current_tracker.slotset
        for s in current_slots:
            s.print_entity()
        inform_slots = [s.inform_slot() for s in current_slots]
        updated_slots = current_tracker.update(previous_slots, inform_slots)
        current_tracker.slotset = updated_slots
        utters = [e.get_text() for e in current_tracker.events] + [q]
        if force_act:
            action_name = force_act
        else:
            print('current intent is {}'.format(self.intent))
            print("last_action",current_tracker.last_action)
            action_name = self.action_model.get_next_action_from_utters_intent_lastact_slot_state(utters,self.intent,current_tracker.last_action,updated_slots)
            print("running ",action_name)
        current_tracker.add_user_event(q, slots=updated_slots)
        i = 0
        action = Navigate(action_name, {},self.reminder_manager)
        action.slots = updated_slots
        result = action.run()
        current_tracker.add_bot_event(result)
        if not result:
            result = ""
        current_tracker.last_action = action_name
        return result,action_name

class Test():
    def __init__(self,a):
        self.agent = a
        self.agent.model_path =  "C://model/pretrain_kvret3_.h5"

    def process(self,q):
        return self.agent.process(q)

    def run(self):
        print("==========    {}   ===========".format(get_task_name_from_config()))
        reply, action = self.process("")
        print('Bot :' + reply)
        while True:
            user_input = input('I:')
            if user_input == "_end":
                break
            reply,action = self.process(user_input)
            print('Bot :'+reply)

def get_agent():
    reminder_manager = ScheduleManager()
    manager = TrackerManager()
    db_manager = DatabaseManger()
    agent = Helper(manager, db_manager,reminder_manager)
    return agent
def start():
    agent = get_agent()
    t = Test(agent)
    t.run()

def test2():
    a = get_agent()
    a.process("set a reminder for me ")
    a.process("tomorrow, 6 pm for my dinner")
    a.process("when is my dinner?")
    print("===================")
    a.process("#clear")
    a.process("when is my dinner?")
