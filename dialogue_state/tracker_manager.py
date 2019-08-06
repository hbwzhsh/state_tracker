from dialogue_state.state_tracker import Tracker

class TrackerManager():
    def __init__(self,id = "test_bot"):
        self.trackers = dict()
        self.id  = id

    def set_tracker_id(self,id):
        '''
        set the tracker id when  server runs
        :param id:tracker id
        :return:None
        '''
        self.id = id

    def add_or_get_tracker(self,id):
        '''
        get the tracker for user id ,
        if the id does not exist
        create a  new tracker for the user
        :param id: user id
        :return: the tracker for the user
        '''
        if not id:
            return
        if id not in self.trackers.keys():
            self.trackers[id] = Tracker(id)
        return self.trackers[id]

    def add_or_get_default_tracker(self):
        return self.add_or_get_tracker("test_bot")

    def current_tracker(self):
        return self.add_or_get_tracker(self.id)




