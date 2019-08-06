import time
from dialogue_state.slot_tool import SlotTool


class Event():
    def __init__(self,type = '',text = '',slotset=None,is_message = True,action_name=None):
        if not type:
            type = ''
        if not text:
            text = ''
        if not slotset:
            slotset = []
        self.type = type
        self.data = dict()
        self.slotset = slotset
        self.is_message = is_message
        if self.type == "user" or "bot":

            self.set_text(text)
            if self.type =="bot":
                self.action_name = action_name

    def set_is_message(self,is_message):
        self.is_message = is_message

    def is_normal(self):
        return self.is_message

    def is_user_intent_confirm(self):
        return self.get_intent().startswith("confirm_intent_")

    def is_bot_intent_confirm(self):
        return self.get_action_name().startswith("bot_confirm_intent_")

    def is_user_slot_confirm(self):
        return self.get_intent().startswith("confirm_slot_")

    def is_bot_slot_confirm(self):
        return self.get_action_name().startswith("bot_confirm_slot_")


    def is_system(self):
        return not self.is_normal()

    def get_action_name(self):
        if self.is_user():
            raise TypeError("expect the bot type event.")
        return self.action_name

    def is_a_message(self):
        return self.is_message

    def set_intent(self,user_intent):
        self.data['user_intent'] = user_intent

    def get_intent(self):
        return self.data.get("user_intent")

    def set_parse_data(self,parse_report):
        self.data['parse_data'] = parse_report

    def get_parse_data(self):

        return self.data.get("parse_data")

    def default_parse_data(self):
        default_return = {"intent": {"name": "", "confidence": 0.0},
                          "entities": [], "text": ""}
        return default_return

    def set_text(self,text):
        self.data['text'] = text

    def get_text(self):
        return self.data['text']

    def is_user(self):
        if not self.type:
            print("event type is ??")
            return False
        return self.type == "user"

    def is_bot(self):
        if not self.type:
            print("event type is ??")
            return False
        return self.type == "bot"

    def set_slotset(self,slots):
        self.slotset = slots

    def get_slotset(self):
        return self.slotset

    def event2dict(self,id = 'testbot'):
        """
        convert event to dict
        :rtype: dict
        """
        d = dict()
        d['id'] = id
        d['type'] = self.type
        d['text'] = self.get_text()
        d['slots'] = SlotTool.slotset2dict(self.get_slotset())
        d['timestamp'] = time.time()
        return d

    @staticmethod
    def dict2event(d):
        type = d.get('type')
        text = d.get('text')
        slotset_d = d.get('slots')
        slotset = SlotTool.dict2slotset(slotset_d)
        e = Event(type,text,slotset)
        return e

    @staticmethod
    def sample_dict(n = 0):
        return {
            "id":"123",
            "slots":{
            "area":"north"
            },
            "text":"This is the {}th event".format(n),
            "type":"user",
            "time": time.time()
        }
    def __str__(self):
        if self.is_user():
            return '{} event:{},\nslots:'.format(self.type,self.get_text())+" \n "+SlotTool.slotset2dict(self.get_slotset()).__str__()

        if self.is_bot():
            return '{} event,{}:{},\nslots:'.format(self.type, self.get_action_name(),self.get_text()) + " \n " + SlotTool.slotset2dict(
                self.get_slotset()).__str__()
