from policy.agent import *
from policy.form_policy import *
from server.nlu_client import *

class MemoryAgent(Helper):
    def __init__(self,manager):
        model_name = get_task_name_from_config()
        self.bot_nlu_client = NluClient()
        self.domain = Domain()
        story_path = any_story.format(model_name)
        self.trackers_manager = manager
        self.policy = MemoryPolicy(story_path)
        self.enable_slot_affirm = True
        self.enable_intent_affirm = True
        self.normal_mode = True
        self.input_slot = False
        self.input_intent = False
        self.need2confrimslot = []
        self.print_info = []#["action","template"]
        self.is_start_form_policy = False
        self.form_policy = None
        self.enable_final_fill = False
        self.reminder_manager = ScheduleManager()

    def reset(self):
        self.enable_slot_affirm = True
        self.enable_intent_affirm = True
        self.normal_mode = True
        self.input_slot = False
        self.input_intent = False
        self.need2confrimslot = []
        self.is_start_form_policy = False
        self.form_policy = None
        self.enable_final_fill = False

    def predict_action(self,past_intents):
        #return predict(self.all_stories,past_intents, self.max_history, domain_label=self.intent_list)
        return self.policy.predict_from_default_dict(past_intents)

    def handle_task_complete(self):
        current_tracker = self.default_tracker()
        finish = self.domain.is_finish_a_story_path(self.domain.taskname, current_tracker.get_past_intent())
        if finish:
            current_tracker.clear_event()
            dbot_online_logger.debug("finish the {}".format(self.domain.taskname))

    def handle_unfilled_slots(self,intent_so_far,q):
        current_tracker = self.default_tracker()
        if self.enable_final_fill:
            if self.domain.is_slots_filled(self.domain.taskname)==False:
                if self.is_start_form_policy or self.domain.is_finish_a_story_path(self.domain.taskname,intent_so_far):
                    self.enable_intent_affirm = False
                    self.enable_slot_affirm = False
                    # enter the formpolicy'
                    dbot_online_logger.debug("----is_finish_a_story_path----")
                    if self.is_start_form_policy == False:
                        self.form_policy = FormPolicy.collecting_form_policy()
                        self.form_policy.slots = current_tracker.slotset
                        self.is_start_form_policy = True
                    reply = self.form_policy.get_bot_reply(user_input=q)
                    current_tracker.slotset = self.form_policy.slots
                    return reply,"formpolicy_action"

    def get_max_task_with_p(self,q):
        intent_rsp_data = self.bot_nlu_client.get_intent_classification_info(user_msg_text=q)
        task_label,probabilitys = intent_rsp_data.get("rsp_nlp_intent_data")
        task = task_label[0]
        task = task.replace("__label__","")
        p = probabilitys[0]
        return task,p

    def process(self,q):
        if self.is_need_handle_user_affirm():
            reply,_ = self.handle_user_affirm(q)
            return  reply,_
        current_tracker = self.default_tracker()
        nlu_result = self.domain.nlu_model.predict(q)

        # use client to switch the task
        # if most likely task is not the same as the current one
        # change
        task,p = self.get_max_task_with_p(q)
        if task != self.domain.taskname and p >0.94:
            dbot_online_logger.debug("task switiched from {} to {}".format(self.domain.taskname, task))
            self.domain.taskname = task
            self.policy.change_task(task)
            current_tracker.clear_event()
            self.reset()

        if self.is_need_intent_affirm(nlu_result,q) or self.is_need_slot_affirm(nlu_result):
            reply,_= self.do_bot_affirm(q,nlu_result)
            return reply,_
        current_slots = []
        if self.domain.taskname in ["reminder","weather"]:
            current_slots = self.domain.get_entity(q)
            current_tracker.show_slot_debug()
        else:
            current_slots = self.domain.get_valid_slots_from_nlu_result(nlu_result,q,slot_threshold)
            current_tracker.show_slot_debug()
        previous_slots = current_tracker.slotset
        current_tracker.slot_merge_and_update(current_slots,previous_slots)
        current_tracker.show_slot_debug()
        have_done = current_tracker.have_done_user_intent_affirm()

        print("have done ? intent affirm ",have_done)
        intent = self.domain.nlu_model.rasa_intent_with_domain(q, self.domain.taskname)
        intent_so_far = current_tracker.get_intent_sofar(intent)
        #filled = self.domain.is_slots_filled(self.domain.taskname,current_tracker)
        # dbot_online_logger.debug("fill {}".format(str(filled)))
        # if filled and  (not self.is_start_form_policy) and not(self.domain.is_finish_a_story_path(self.domain.taskname, intent_so_far)):
        #     reply = current_tracker.get_summary_slot_under_task(self.domain.taskname)
        #     current_tracker.clear_event()
        #     return reply, "utter_finish"
        res = self.handle_unfilled_slots(intent_so_far,q)
        if res:
            return res
        action_or_utter_name = self.predict_action(intent_so_far)
        current_tracker.add_user_event(q,parsing_result=nlu_result, slots=previous_slots, intent=intent)
        reply_list = self.actions_update(current_tracker, action_or_utter_name, current_tracker.slotset)
        self.handle_task_complete()
        replys = "\n".join(reply_list)
        return replys,"??"

# manager = TrackerManager()
# agent = MemoryAgent(manager)
# # local test for the agent
# t = Test(agent)
# r,_ = t.process("what is the weather like in beijing?")
# print(r)
#
# r,_ = t.process("Beijing Thursday")
# print(r)
# r,_ = t.process("inform_city")
# print(r)

# will run the loop and keep calling "process()"
#t.run()

