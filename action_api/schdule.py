class ScheduleManager():
    def __init__(self):
        self.events =[]

    def add_event_by_name(self,name):
        e = {"event":name}
        self.events.append(e)
        print("add event %s"%name)

    def add_event(self,e):
        self.events.append(e)

    def sample_event(self):
        return {
                   "room": "-",
                   "agenda": "-",
                   "time": "2 pm",
                   "date": "thursday",
                   "party": "-",
                   "event": "swimming activity"
               }
    def null_event(self):
        return {
                   "room": "-",
                   "agenda": "-",
                   "time": "-",
                   "date": "-",
                   "party": "-",
                   "event": "-"
               }

    def add_new_event(self,room=None,agenda =None,time=None,date=None,party=None,eventname=None):
        event = self.null_event()
        if room:
            event["room"] = room
        if agenda:
            event["agenda"] = agenda
        if time:
            event["time"] = time
        if date:


            event["date"] = date
        if party:
            event["party"] = party
        if event:
            event["event"] = eventname
        self.add_event(event)