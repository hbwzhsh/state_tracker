class System():
    def __int__(self):
        pass

    def is_debug_call(self,q):
        for word in ["intent:","#check_entity","#turns_slot","#undo","#clear","#printevent","_alter"]:
            if word in q:
                return True
        return False



    def system_call(self,q,tracker):
        #tracker = self.trackers_manager.add_or_get_default_tracker()
        if q.startswith('intent:'):
            intent = q.split(":")[1]
            self.set_intent(intent)
            return 'update {}'.format(intent),"_"
        elif q=="#check_entity":
            tracker.check_slot()
            return "check_entity","_"
        elif q=="#turns_slot" or q=="#all_slot":
            tracker.check_slots_for_allturns()
            return "print all turn slots", "_"
        elif q=="#undo":
            tracker.undo()
            #repeat the action /reply
            tracker.repeat_lastturn_replys()
            return "undo this turn","_"
        elif q == "#clear":
            tracker.clear_event()
            return "clear events","_"
        elif q == "#printevent":
            tracker.print_events()
            tracker.repeat_lastturn_replys()
            return "printevent","_"
        elif q.startswith("_alter"):
            for task in ["book_course","find_apartment","order_food"]:
                if task in q:
                    print("alter task {} and clear events".format(task))
                    self.alter_rasa_model(task)
            tracker.clear_event()
            return "alter task {} and clear events".format(task),"_"
        return "unknown system act","_"

    def is_force_act(self,q):
        return q.startswith('-')