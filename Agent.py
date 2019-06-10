from action_model import ActionModel,MultiClassModel,BinaryModel,RnnBinaryModel,Glove
from action import Action
from slot_filling_model import SlotModel,HelperNER
from state_tracker import Tracker
from task_action import Task,Navigate
from event import Event
from entity import Entity

from Datebase import DatabaseManger
from TrackerManager import TrackerManager

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
        self.trackers = manager

    def process(self,q):
        parsing_result = self.slot_model.predict(q)
        current_tracker = self.trackers.add_or_get_tracker('test_bot')
        user_event = current_tracker.add_user_event(q, parsing_result)
        if self.db_manager:
            self.db_manager.insert_event_mongo(user_event.event2dict('test_bot'))

        slu = self.slot_model.rasa2slu(parsing_result)
        previous_slots = current_tracker.slotset
        current_slots = current_tracker.rasa_to_slots(parsing_result)
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
    def  __init__(self,manager = None ,db_manager = None):
        """
        init and load model here，rasa model for slot filling ,keras basic dense multi label classifcatioin for multi-action
        """
        self.slot_model = HelperNER()
        self.action_model = Glove()
        #self.model_path = "C://model/pretrain_kvret_.h5"
        self.model_path = "C://model/pretrain_kvret2_.h5"

        self.action_model.all_action = self.action_model.kvret_actions()
        self.db_manager  = db_manager
        #self.action_model.load_model(self.action_model.model_path)
        self.trackers = manager
        self.action_model.load_model(self.model_path)
        self.inform_slot = ['location','weather_attribute','date','time','date','event']
        self.intent = "weather"

    def set_intent(self,intent):
        self.intent = intent

    def get_action_query(self,q):
        # _actionxxx:query
        if ":" not in q:
            raise ValueError(" \":\" should be inside the query but query is {}".format(q))

        action = q.split(":")[0][1:]
        query = q.split(":")[1]
        return action,query

    def process(self,q):
        force_act = None

        if q.startswith('_'):
            force_act,query = self.get_action_query(q)
        elif q.startswith('intent:'):
            intent = q.split(":")[1]
            self.set_intent(intent)
            return 'update {}'.format(intent),"_"
        elif q == "#clear":
            self.trackers.add_or_get_tracker('test_bot').clear_event()
            return "clear events","_"
        if force_act:
            q = query
        current_slots = self.slot_model.get_entity(q)
        current_tracker = self.trackers.add_or_get_tracker('test_bot')
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
            #action_name = self.action_model.get_next_action_from_utters_intent(utters,self.intent)
            print("action",current_tracker.last_action)
            print("utters",utters)
            action_name = self.action_model.get_next_action_from_utters_intent_lastact(utters,self.intent,current_tracker.last_action)
            print("running ",action_name)
        current_tracker.add_user_event(q, slots=updated_slots)
        i = 0
        action = Navigate(action_name, {})
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
        self.agent.model_path =  "C://model/pretrain_kvret_.h5"

    def process(self,q):
        return self.agent.process(q)

    def run(self):
        while True:
            user_input = input('I:')
            if user_input == "_end":
                break
            reply,action = self.process(user_input)
            print('Bot :'+reply)


# test
manager = TrackerManager()
db_manager = DatabaseManger()
# # # agent = Agent(manager,db_manager)
# # #
# # # r = 1
# # #
# # # agent.process('I want chinese food')
# # # agent.process('Can I have the address?')
# # # agent.process('address?')
# #
agent = Helper(manager,db_manager)
t = Test(agent)
t.run()

# print(agent.process('where\'s the nearest parking garage','navigate'))
# agent.trackers.add_or_get_tracker('test_id').clear_event()
# print(agent.process('help me set up a reminder'))
#print(agent.process('Whats the two day forecast for new york, does it say it will be hot?'))
# i = 2