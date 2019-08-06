from typing import List
from dialogue_state.entity import Entity
from dialogue_state.event import Event
from loggers import dbot_online_logger


class Tracker():
    def __init__(self, id='guest'):
        self.id = id
        self.name = 'Tracker_' + self.id
        self.events = []
        self.latest_message = ""
        self.slotset = []
        self.last_action = ""

    def add_user_event(self, q, parsing_result=None, slots=None,intent = None):
        """
        add one user type normal query event into the tracker events
        :param q: user query/input
        :param parsing_result: the rasa nlu parsing result or other equivalent format
        :param slots:the list of slot[entity]
        :param intent:str
        :return:the event
        """
        print("add user event ,intent",intent)
        dbot_online_logger.debug("the user say %s" %q)
        dbot_online_logger.debug("add user event ,intent {}".format(intent))
        event = Event('user', q)
        self.latest_message = q
        if parsing_result:
            event.set_parse_data(parsing_result)
            event.set_slotset(self.rasa_to_slots(parsing_result))
        if slots:
            event.set_slotset(slots)
        # replace by other format-converter
        if intent:
            event.set_intent(intent)
        self.add_event(event)
        return event

    def add_user_confirm(self,q,intent,slots):
        """
        add one user type confirm  event into the tracker events
        :param q: user query/input
        :param slots:the list of slot[entity]
        :param intent:str
        :return:the event
        """
        print("add_user_confirm ,intent",intent)
        dbot_online_logger.debug("the user say %s" % q)
        dbot_online_logger.debug("add add_user_confirm ,intent {}".format(intent))
        event = Event('user', q,is_message=False)
        self.latest_message = q
        if slots:
            event.set_slotset(slots)
        # replace by other format-converter
        if intent:
            event.set_intent(intent)
        self.add_event(event)
        return event


    def add_bot_event(self, r,slotset=None,action_name = None):
        """
        add one user type confirm  event into the tracker events
        :param r: bot reply or other type of text response
        :param slotset:the list of slot[entity] at this turn
        :param action_name:str,the action name of the bot /the utter template of the bot
        :return:the event
        """
        event = Event(type='bot', text=r)
        if slotset:
            event.slotset = slotset
        if action_name:
            event.action_name = action_name
        self.add_event(event)

    def add_bot_confirm(self, r,slotset=None,action_name = None):
        event = Event(type='bot', text=r)
        if slotset:
            event.slotset = slotset
        if action_name:
            event.action_name = action_name
        self.add_event(event)

    def have_done_user_intent_affirm(self):
        """
        get access to history of events and check if there is user affrim event
        :return:True or False
        """
        index = None
        done  = False
        for i,event in enumerate(self.events):
            if event.is_user() and event.is_a_message():
                index = i
            if event.is_bot() and event.is_a_message():
                index = i
        for i,event in enumerate(self.events):
            if i >=index:
                if event.get_intent():
                    if event.get_intent().startswith("confirm_intent_"):
                        done = True
        return done

    def last_action_is_normal(self):
        """
        get access to history of events and check if whether the last action event is normal or affirm
        :return:True or False
        """
        events = [event for event in self.events if event.is_bot()]
        if len(events) ==0:
            print("no bot action found in last action ")
            return False
        last_action = events[-1]
        return last_action.is_normal()

    def last_action_is_intent_affirm(self):
        """
        get access to history of events and check if whether the last user event is affirm
        :return:True or False
        """
        events: List[Event] = [event for event in self.events if event.is_bot()]
        if len(events) == 0:
            print("no bot action found in last action ")
            return False
        last_action = events[-1]
        return last_action.is_bot_intent_confirm()

    def last_action_is_slot_affirm(self):
        """
        get access to history of events and check if whether the last user event is slot affirm
        :return:True or False
        """
        events: List[Event] = [event for event in self.events if event.is_bot()]
        if len(events) == 0:
            print("no bot action found in last action ")
            return False
        last_action = events[-1]
        return last_action.is_bot_slot_confirm()

    def is_user_intent_affirm(self):
        """
        whether user should do intent affirm in this turn
        :return:True or False
        """
        return self.last_action_is_intent_affirm()


    def is_user_slot_affirm(self):
        """
        whether user should do slot affirm in this turn
        :return:True or False
        """
        return self.last_action_is_slot_affirm()



    def is_normal_user_query(self):
        """
        use this method before insert the new user event
        :return: T F for need to nlu process
        """
        return self.last_action_is_normal()

    def have_done_user_slot_affirm(self):
        """
        get access to history of events and check if whether user have affirmed in this turn
        :return: T F for need to nlu process
        """
        index = None
        done  = False
        for i,event in enumerate(self.events):
            if event.is_user() and event.is_a_message():
                index = i
            if event.is_bot() and event.is_a_message():
                index = i
        for i,event in enumerate(self.events):
            if i >=index:
                if event.get_intent():
                    if event.get_intent().startswith("confirm_slot_"):
                        done = True
        return done

    def add_event(self, event):
        """
        add a new event to the events

        """
        self.events.append(event)

    def add_muti_events(self, events):
        """
        add mutiple events

        """
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

    def print_events(self):
        for e in self.events:
            if e.is_user():
                print("----------------------------------")
            print(e)

    def repeat_lastturn_replys(self):
        index = self.get_last_user_event_index()
        for i, e in enumerate(self.events):
            if i > index and e.is_bot():
                print("Bot :{}".format(e.get_text()))

    def get_last_user_event_index(self):
        undo_index = None
        for i,event in enumerate(self.events):
            if event.type == "user":
                undo_index = i
        return undo_index

    def undo(self):
        '''
        system debug command for undo this turn ,delete the events generated at the turn
        :return:
        '''
        undo_index= self.get_last_user_event_index()
        if undo_index:
            self.events = self.events[:undo_index]
            self.slotset = self.events[-1].slotset
        print("[SYSTEM act] undo done!")
        #last_event = self.events[-1]
        #last_event.set_slotset(self.rasa_to_slots(last_event.get_parse_data()))
        return

    def check_slots_for_allturns(self):
        '''
        show slots turn by turn
        :return:
        '''
        turn_i = 0
        for event in self.events:
            if event.is_user():
                print("turn ",turn_i)
                turn_i+=1
                if not event.slotset or event.slotset == []:
                    print("empty slot.")
                for e in event.slotset:
                    print("[info] current slot,", e.get_entity_type(), e.get_entity_value())
                print("-------")
            if event.is_bot():
                if not event.slotset or event.slotset == []:
                    print("empty slot.")
                for e in event.slotset:
                    print("[info] current slot,", e.get_entity_type(), e.get_entity_value())
                print("-------")
        return

    def get_summary_slot_under_task(self,taskname):
        '''
        get the required slot names under the task
        and check if we have collected all the slots
        :param taskname: task /story name
        :return: True or False
        '''
        required_slot_names = get_required_slot_names_by_task(taskname)
        if not self.slotset or self.slotset==[]:
            print("empty slot.")
            return "empty slot."
        res = ""
        for slot in required_slot_names:
            for e in self.slotset:
                if slot == e.get_entity_type():
                    res+="[summary] current slot,{} {}".format(e.get_entity_type(),e.get_entity_value())
                    res +="\n"
                t = e.cut_inform()
                if slot == t.get_entity_type():
                    res+="[summary] current slot,{} {}".format(t.get_entity_type(),t.get_entity_value())
                    res +="\n"
        return res

    def check_slot(self):
        '''
        show all slots
        :return: slots
        '''
        if not self.slotset or self.slotset==[]:
            print("empty slot.")
            return "empty slot."
        res = ""
        for e in self.slotset:
            res+="[info] current slot,{} {}".format(e.get_entity_type(),e.get_entity_value())
            res +="\n"
        return res

    def rasa_to_slots(self, parsing_result,threshold = 0):
        '''
        rasa nlu model parsing result -> slot list
        :param parsing_result: rasa nlu model parsing result
        :param threshold: minimum confidence for allowing take the slot
        :return: slot list
        '''
        if not parsing_result:
            print("no rasa parsing result found")
            return []
        entities = parsing_result['entities']
        slots = []
        for e_dict in entities:
            user_act_slot = e_dict['entity']
            slot_value = e_dict['value']
            confidence = e_dict['confidence']
            if confidence <threshold:
                continue
            else:
                #print("from tracker :added the high confidence slot {},{},{}".format(user_act_slot,slot_value,confidence))
                slot = Entity(user_act_slot)
                slot.set_value(slot_value)
                slots.append(slot)
        return slots

    def add_inform(self, slots, names):
        for s in slots:
            if s.type_name not in names: continue
            s.type_name = 'inform_' + s.type_name
        return slots

    def cut_inform(self, slots, names):
        """
        get the non_prefix slot name
        :param slots: slot list
        :param names: slotname to be cut
        :return: updated slots
        """
        for s in slots:
            if s.type_name[7:] not in names: continue
            s.type_name = s.type_name.replace('inform_', '')
        return slots

    def update(self, slots1, slots2):
        """
        merge the old slots with current slots
        :param slots1: previous slots
        :param slots2: current slots for this turn
        :return: combined slots
        """
        d = dict()
        for slot in slots1 + slots2:
            if slot.type_name.startswith('inform'):
                if slot.enable_extend:
                    print(slot.type_name,"is extend")
                    v = slot.get_entity_value()
                    if slot.type_name not in d.keys():
                        print(slot.type_name,"not in the key",d.keys())
                        if not isinstance(v, list):
                            d[slot.type_name] = [slot.get_entity_value()]
                            print("add [],first slot",slot.get_entity_type())
                        else:
                            d[slot.type_name] = v
                            print("first slot", slot.get_entity_type())
                    else:
                        print(slot.type_name, " in the key", d.keys())
                        if not isinstance( d[slot.type_name], list):
                            d[slot.type_name] = [ d[slot.type_name]]
                            print("slot value to []")
                        if not isinstance(v,list):
                            d[slot.type_name] = d[slot.type_name] + [v]
                            print("add [],sec slot", slot.get_entity_type())
                        else :
                            d[slot.type_name] = d[slot.type_name] + v
                            print("sec slot", slot.get_entity_type())

                else:
                    print(slot.type_name, "is not extend")
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

    def show_slot_debug(self):
        '''
        logger debug to show the slots collected
        :return:
        '''
        dbot_online_logger.debug("----slots----")
        for slot in self.slotset:
            dbot_online_logger.debug("slots:{}".format(slot.__str__()))

    def slot_from_events(self, events):
        return self.slot_from_history([e.get_slotset() for e in events])

    def update_and_set_slot(self, newslot):
        self.slotset = self.update(self.slotset, newslot)

    def get_filled_slot_names(self):
        slots = [u.cut_inform() for u in self.slotset]
        return [slot.get_entity_type() for slot in slots]

    def slot_merge_and_update(self,current_slots,previous_slots):
        """
        update slots for tracker
        """
        if current_slots == None:
            current_slots = []
        inform_slots = [s.inform_slot() for s in current_slots]
        updated_slots = self.update(previous_slots, inform_slots)
        self.slotset = updated_slots

    def get_last_nlu_result(self):
        nlu_res = None
        for e in self.events:
            if e.type =="user":
                tmp = e.get_parse_data()
                if tmp and tmp!="" and tmp!=[]:
                    nlu_res = tmp
        if not nlu_res:
            print("not nlu result for rasa is found.")
        return nlu_res

    def get_past_intent(self):
        intents =[]
        for e in self.events:
            if e.type =="user":
                if e.get_intent().startswith("confirm_intent_"):
                    intents.append(e.get_intent().replace("confirm_intent_",""))
                elif e.get_intent().startswith("confirm_slot_"):
                    continue
                elif e.get_intent()!="undefined" and e.get_intent()!="undefine":
                    intents.append(e.get_intent())
        return intents
        #
        # past_intents = [e.get_intent() for e in self.events if e.type == "user"and e.get_intent()!="undefine"]
        # return past_intents

    def get_intent_sofar(self,current_intent):
        past_intents = self.get_past_intent()+ [current_intent]
        #print("so far , intent are {}".format("|".join(past_intents)))
        return past_intents

    def get_last_intent(self):
        #past_intents = [e.get_intent() for e in self.events if e.type == "user"]
        return  self.get_past_intent()[-1]


