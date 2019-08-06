import pymongo
from pymongo import MongoClient

from dialogue_state.event import Event


class DatabaseManger():
    def __init__(self):
        self.collection = self.get_data_coll()

    def get_data_coll(self):
        client = MongoClient('mongodb://sbdrg:Aa123654@10.255.0.3:27017/dialogue_bot')
        coll = client['dialogue_bot']['events']
        return coll

    # def insert_event_mongo(self,event_dict):
    #     self.collection.insert_one(event_dict)

    def insert_event_mongo(self, event_dict):
        bot_id = event_dict['id']
        doc = self.collection.find_one({"id":bot_id})
        if not doc or 'events' not in doc.keys():
            self.collection.insert_one({'id':bot_id,'events':[event_dict]})
            return
        else:
            doc['events'].append(event_dict)
            self.del_by_id(doc['_id'])
            # find one
            # append
            # replace
            self.collection.insert_one(doc)

    def clear_events(self, bot_id):
        self.collection.remove({"id":bot_id})

    def del_by_id(self,id):
        self.collection.remove({"_id":id})

    def del_last_event(self,bot_id):
        res = []
        for u in self.collection.find({"id":bot_id}).sort([
                    ('time', pymongo.ASCENDING)]):
            res.append(u)
            print(u)
        first = res[0]
        last = res[-1]
        self.del_by_id(last['_id'])

    def del_first_event(self,bot_id):
        res = []
        for u in self.collection.find({"id":bot_id}).sort([
                    ('time', pymongo.ASCENDING)]):
            res.append(u)
            print(u)
        first = res[0]
        last = res[-1]
        self.del_by_id(first['_id'])

    def get_events_by_id(self,bot_id):
        res = []
        for u in self.collection.find({"id":bot_id}).sort([
                    ('time', pymongo.ASCENDING)]):
            res.append(u)
        events = [Event.dict2event(d) for d in res]
        return events

    def restore_events_from_mongo(self,bot_id):
        doc = self.collection.find_one({"id":bot_id})
        return doc.get('event')






# coll = get_data_coll()
# a = coll.insert_one({'a': [1]})

# manager = DatabaseManger()
# for i in range(5):
#     manager.insert_event_mongo(Event.sample_dict(i))
# #manager.clear_events('123')
# manager.del_first_event("123")
