from action_model import ActionModel,MultiClassModel,BinaryModel,RnnBinaryModel
from action import Action
from slot_filling_model import SlotModel,HelperNER
from state_tracker import Tracker
from task_action import Task,Navigate
from entity import Entity
class Agent():
    def __init__(self,manager):
        """
        init and load model here，rasa model for slot filling ,keras basic dense multi label classifcatioin for multi-action
        """
        camb_rest_slot_path = 'C:\\Users\\Administrator\\PycharmProjects\\Dialogue_Bot\\models\\current\\CamRest_slot_filling'
        self.slot_model = SlotModel()
        self.slot_model.load(camb_rest_slot_path)
        self.action_model = RnnBinaryModel()
        self.action_model.load_models()
        #self.action_model.load_model(self.action_model.model_path)
        self.trackers = manager

    def process(self,q):
        parsing_result = self.slot_model.predict(q)
        current_tracker = self.trackers.add_or_get_tracker('test_bot')
        current_tracker.add_user_event(q, parsing_result)
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