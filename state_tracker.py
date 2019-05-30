from entity import Entity
from event import Event


class Tracker():
    def __init__(self, id='guest'):
        self.id = id
        self.name = 'Tracker_' + self.id
        self.events = []
        self.latest_message = ""
        self.slotset = []
        self.last_action = ""

    def add_user_event(self, q, parsing_result=None, slots=None):
        event = Event('user', q)
        self.latest_message = q
        if parsing_result:
            event.set_parse_data(parsing_result)
            event.set_slotset(self.rasa_to_slots(parsing_result))
        if slots:
            event.set_slotset(slots)
        # replace by other format-converter
        self.add_event(event)
        return event

    def add_bot_event(self, r):
        event = Event('bot', r)
        self.add_event(event)

    def add_event(self, event):
        self.events.append(event)

    def add_muti_events(self, events):
        self.events.extend(events)

    def fill_slot_by_entity(self, entitys, slots):
        slot_list = []
        for entity in entitys:
            if entity.get_entity_type() in slots:
                slot_list.append(entity)
        return slot_list

    def __str__(self):
        s = ''
        for e in self.events:
            s += (e.__str__() + '\n')
        return s

    def clear_slots(self):
        self.slotset = []

    def clear_latest_message(self):
        self.latest_message = ""

    def clear_last_action(self):
        self.last_action = ""

    def clear_event(self):
        self.events = []
        self.clear_last_action()
        self.clear_latest_message()
        self.clear_slots()

    def rasa_to_slots(self, parsing_result):
        entities = parsing_result['entities']
        slots = []
        for e_dict in entities:
            user_act_slot = e_dict['entity']
            slot_value = e_dict['value']
            if user_act_slot.split('_')[0] == 'inform':
                slot = Entity(user_act_slot)
                slot.set_value(slot_value)
                slots.append(slot)
            if user_act_slot.split('_')[0] == 'request':
                ua = user_act_slot.split('_')[0] + '_' + slot_value
                slot = Entity(ua)
                slots.append(slot)
        return slots

    def add_inform(self, slots, names):
        for s in slots:
            if s.type_name not in names: continue
            s.type_name = 'inform_' + s.type_name
        return slots

    def cut_inform(self, slots, names):
        for s in slots:
            if s.type_name[7:] not in names: continue
            s.type_name = s.type_name.replace('inform_', '')
        return slots

    def update(self, slots1, slots2):
        d = dict()
        for slot in slots1 + slots2:
            if slot.type_name.startswith('inform'):
                d[slot.type_name] = slot.get_entity_value()
        reset = False
        for slot in slots2:
            if slot.type_name.startswith('request'):
                d[slot.type_name] = slot.get_entity_value()
                reset = True
        if not reset:
            for slot in slots1:
                if slot.type_name.startswith('request'):
                    d[slot.type_name] = slot.get_entity_value()
        new = []
        for k, v in d.items():
            slot = Entity(k)
            slot.set_value(v)
            new.append(slot)
        return new

    def slot_from_history(self, slot_hs):
        if len(slot_hs) <= 1:
            return slot_hs
        else:
            tmp = slot_hs[0]
            for i, slots in enumerate(slot_hs):
                if i == 0: continue
                tmp = self.update(tmp, slots)
            return tmp

    def slot_from_events(self, events):
        return self.slot_from_history([e.get_slotset() for e in events])

    def update_and_set_slot(self, newslot):
        self.slotset = self.update(self.slotset, newslot)
