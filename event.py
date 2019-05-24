from slot_tool import SlotTool
class Event():
    def __init__(self,type = '',text = '',slotset=None):
        self.type = type
        self.data = dict()
        self.slotset = slotset
        if self.type == "user" or "bot":
            self.set_text(text)

    def set_parse_data(self,parse_report):
        self.data['parse_data'] = parse_report

    def get_parse_data(self):
        return self.data['parse_data']

    def default_parse_data(self):
        default_return = {"intent": {"name": "", "confidence": 0.0},
                          "entities": [], "text": ""}
        return default_return

    def set_text(self,text):
        self.data['text'] = text

    def get_text(self):
        return self.data['text']

    def set_slotset(self,slots):
        self.slotset = slots

    def get_slotset(self):
        return self.slotset

    def event2dict(self):
        d = dict()
        d['type'] = self.type
        d['text'] = self.get_text()
        d['slots'] = SlotTool.slotset2dict(self.get_slotset())
        return d

    @staticmethod
    def dict2event(d):
        type = d.get('type')
        text = d.get('text')
        slotset_d = d.get('slots')
        slotset = SlotTool.dict2slotset(slotset_d)
        e = Event(type,text,slotset)
        return e

    def __str__(self):
        return '{} event:{}'.format(self.type,self.get_text())
