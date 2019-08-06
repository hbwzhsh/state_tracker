from policy.agent import *
from policy.fsm import Fsm
from server.nlu_client import *
# 导入模块
class MachineAgent(Agent):
    def __init__(self,manager=None):
        self.bot_nlu_client = NluClient()
        self.domain = Domain()
        self.trackers_manager = manager
        self.fsms = {
            'buy_tv':Fsm('buy_tv'),
            'camrest':Fsm('camrest'),
            'weather':Fsm('weather'),
            'reminder':Fsm('reminder')
                     }
        self.fsm = self.fsms.get('weather')#self.fsms.get(self.domain.taskname)
        self.reminder_manager = ScheduleManager()
        self.first_start = True

    def predict_action(self, intent):
        '''
        :param intent:
        :return: list[str],the list of utter name / action name
        '''
        return self.fsm.predict(intent)

    def reset(self):
        if self.fsm:
            self.fsm.return_start_node()
        self.domain.set_slots([])
        self.domain.set_turn_intent(None)

    def get_max_task_with_p(self,q):
        #return "buy_tv",0.999
        # intent_rsp_data = self.bot_nlu_client.get_intent_classification_info(user_msg_text=q)
        # task_label,probabilitys = intent_rsp_data.get("rsp_nlp_intent_data")
        # max_p = 0
        # max_task = ""
        # dbot_online_logger.debug(task_label)
        # dbot_online_logger.debug([probabilitys])
        # for task ,p  in zip(task_label,probabilitys):
        #     #task = task_label[0]
        #     task = task.replace("__label__","")
        #     #p = probabilitys[0]
        #     if task in ["reminder","weather","buy_tv"]:
        #         max_task = task
        #         max_p = p
        #         break
        #
        # return max_task,max_p
        return "weather",1.0

    def has_weekday(self,q):
        if not q:
            return False
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for day in weekdays:
            if day in q:
                return True
        return False

    def process(self,q):
        q = q.strip(" ")
        self.domain.set_current_user_query( q)
        current_tracker = self.default_tracker()
        task, p = self.get_max_task_with_p(q)
        if self.fsm:
            if not self.fsm.is_start_node():
                p = p * 0.99
        dbot_online_logger.debug("domain classification result {}:{}.".format(task,str(p)))
        q_len = len(q.split(" "))
        if  (q_len <3 and self.has_weekday(q)):
            dbot_online_logger.debug("q length is {},has weekday is {}".format(q_len,self.has_weekday(q)))
            dbot_online_logger.debug(" so dont do domain classification and task switich")
        else:

            dbot_online_logger.debug("q")
            dbot_online_logger.debug("len q is {}".format(len(q.split(" "))))
            if self.domain.taskname == None or (task != self.domain.taskname and p >0.80):
                dbot_online_logger.debug("task switiched from {} to {}".format(self.domain.taskname, task))
                self.domain.taskname = task
                self.fsm = self.fsms.get(self.domain.taskname)
                current_tracker.clear_event()
                self.reset()
        if not self.fsm:
            return "sry I do not under your query.you can say buy tv/set a reminder for me /what is the weather like.","undefined"
        if self.fsm:
            if self.fsm.is_current_state_end():
                print("return start")
                self.fsm.return_start_node()
            if self.fsm.is_start_node():
                current_tracker.clear_event()
                self.domain.set_slots([])
                self.domain.set_turn_intent(None)
        intent = self.domain.nlu_model.rasa_intent_with_domain(q, self.domain.taskname)
        self.domain.set_turn_intent(intent)
        nlu_result = self.domain.nlu_model.predict(q)
        current_slots = []
        if self.domain.taskname in ["reminder","weather"] or not self.domain.taskname:

            current_slots = self.domain.get_entity(q)
        elif self.domain.taskname == "camrest":
            sf = CamRestSlotFilling()
            current_slots =sf.find_all_types_slot(q)
        else:
            current_slots = self.domain.get_valid_slots_from_nlu_result(nlu_result,q,slot_threshold)
        previous_slots = current_tracker.slotset
        current_tracker.slot_merge_and_update(current_slots,previous_slots)

        self.domain.set_slots(current_tracker.slotset)
        current_tracker.show_slot_debug()
        if self.fsm:
            condition = self.fsm.get_matched_condition_edge(self.domain)
            dbot_online_logger.debug('--------there is one  matched fsm_edge_condition.'.format(condition))
            #action_or_utter_name = self.predict_action(intent)
            action_or_utter_name = self.fsm.predict(condition)
            dbot_online_logger.debug("actions are ({})".format(action_or_utter_name))
        current_tracker.add_user_event(q, parsing_result=nlu_result, slots=current_tracker.slotset, intent=intent)
        reply_list = self.actions_update(self.default_tracker(), action_or_utter_name, self.default_tracker().slotset)
        replys = "\n".join(reply_list)
        dbot_online_logger.debug("reply is [{}]".format(replys))
        return replys,"??"

def test_agent():
    manager = TrackerManager()
    agent = MachineAgent(manager)
    # local test for the agent
    t = Test(agent)
    # will run the loop and keep calling "process()"
    t.run()

test_agent()