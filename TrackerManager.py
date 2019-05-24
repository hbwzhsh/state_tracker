from state_tracker import Tracker
class TrackerManager():
    def __init__(self):
        self.trackers = dict()

    def add_or_get_tracker(self,id):
        if not id:
            return
        if id not in self.trackers.keys():
            self.trackers[id] = Tracker(id)
        return self.trackers[id]

