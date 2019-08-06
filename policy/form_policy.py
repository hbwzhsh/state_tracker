from dialogue_state.domain import *
from dialogue_state.task_action import *

def find_slot_by_type(slots, slot_type):
    if slots:
        for slot in slots:
            if slot.get_entity_type() == slot_type:
                return slot
            if slot.get_entity_type() == 'inform_' + slot_type:
                return slot.cut_inform()
    return None

def get_slots(slots,required):
    d = dict()
    for slot_name in required:
        d[slot_name] = find_slot_by_type(slots,slot_name)
    lack_slot_names = [k for k,v in d.items() if v == None]
    print("------------------lack slots {}".format(",".join(lack_slot_names)))
    return lack_slot_names,d

def utter_get_next_slot(lack_slot_names):
    if lack_slot_names==[]:
        return
    list_utter = Navigate.get_tmp_by_action("utter_list_"+lack_slot_names[0])
    ask_utter = Navigate.get_tmp_by_action("utter_ask_"+lack_slot_names[0])
    if list_utter == None:
        list_utter = ""
    return list_utter+"\n"+ask_utter

    #return "please input the {}".format(lack_slot_names[0])

def raw_fill(answer,slot_name):
    return Entity(slot_name,answer)

def ask_to_confirm(slots):
    s = ""
    s += "The input slots are summarised below:\n"
    for slot in slots:
        s+="{}:{} \n".format(slot.get_entity_type(),slot.get_entity_value())
    s += "do you comfirm? yes/no\n"
    s += "\n"
    return s

def ask_for_correct(slot_names):
    s = ""
    s +="which is the wrong slot value?\n"
    for i,slot in enumerate(slot_names):
        s += "{},{}\n".format(i,slot)
    return s

class FormPolicy():
    def __init__(self,slot_name):
        self.state = "start"
        self.require_slot_names = slot_name
        self.slots = []
        self.input_slot = None

    @staticmethod
    def collecting_form_policy():
        task_name = get_task_name_from_config()
        require_slots = get_required_slot_names_by_task(task_name)
        require_slots = ["inform_" + u for u in require_slots]
        form_policy = FormPolicy(require_slots)
        form_policy.state = "collecting"
        return form_policy

    def set_slots(self,slots):
        self.slots = slots

    def ask_next_unfilled_slot(self,lack_slot_names):
        """
        take current unfilled slot name
        find the utterance template by the first unfilled slot name
        :rtype: str
        """
        if lack_slot_names == []:
            return
        self.input_slot = lack_slot_names[0]
        list_utter = Navigate.get_tmp_by_action("utter_list_" + lack_slot_names[0])
        ask_utter = Navigate.get_tmp_by_action("utter_ask_" + lack_slot_names[0])
        if list_utter == None:
            list_utter = ""
        return list_utter + "\n" + ask_utter


    def get_bot_reply(self, user_input):
        """
        set or change current state
        produce bot reply according to the user_input
        :rtype: str
        """
        if self.state == "start":
            self.state = "collecting"
            lack, lack_num = get_slots(self.slots,self.require_slot_names)
            return self.ask_next_unfilled_slot(lack)
        if self.state == "collecting":
            if self.input_slot !=None:
                e = Entity(self.input_slot, user_input)
                self.slots.append(e)
            lack,current_slot_dict =  get_slots(self.slots,self.require_slot_names)
            if len(lack)!= 0 :
                dbot_online_logger.debug("----collecting not finish----")
                print(lack,current_slot_dict)
                return self.ask_next_unfilled_slot(lack)
            elif  len(lack) == 0:
                dbot_online_logger.debug("----collecting finish ----")
                self.state = "confirming"
                return ask_to_confirm(self.slots)
        elif self.state == "confirmed":
            return "all slot filled"
        if user_input == "yes" and self.state =="confirming":
            print("you just say yes for confirm the form input")
            return "all slot filled"
        if user_input == "no"and self.state =="confirming":
            self.state = "ask_for_correct"
            return ask_for_correct(self.require_slot_names)
        if user_input == "no" and self.state == "anything else":
            return "all slot filled"
        if user_input == "yes" and self.state == "anything else":
            self.state = "ask_for_correct"
        if self.state == "ask_for_correct" or "anything else":
            slotname,value = user_input.split("=")
            s = find_slot_by_type(self.slots, slotname)
            s.set_value(value)
            self.state = "anything else"
            return "Is there other mistake?\n"+ask_for_correct(self.require_slot_names)

def test():
    p  = FormPolicy(["size","brand","resolution"])
    print("User say:Hi")
    print("Bot say:{}".format(p.get_bot_reply("Hi")))
    while True:
        user_q = input("User say:")
        print("Bot say:{}".format(p.get_bot_reply(user_q)))
#test()
