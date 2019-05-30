from sanic import Sanic
from sanic.response import json,text
from entity import Entity
from TrackerManager import TrackerManager
from slot_tool import SlotTool
from event import Event
from Datebase import DatabaseManger
from Agent import Agent
from sanic_cors import CORS
app = Sanic()
CORS(app)
manager = TrackerManager()
db_manager = DatabaseManger()
agent = Agent(manager,db_manager)

@app.route("/test",methods=['GET','POST','OPTIONS'])
async def test(request):
    return json({"hello": "world"})

def get_sample_slot():
    es = []
    ts = ['food','area']
    vs = ['chinese','north']
    for entity_type,v in zip(ts,vs):
        e = Entity(entity_type,v)
        es.append(e)
    return es

@app.route("/event/add",methods=['GET','POST','OPTIONS'])
async def add_event(request):
    print(request.args)
    print(request.form)
    tracker_id = request.json.get('id')
    slot_d = request.json.get('slots')
    type = request.json.get('type')
    text = request.json.get('text')
    if slot_d:
        slotset = SlotTool.dict2slotset(slot_d)
    else:
        slotset = []
    event = Event(type,text,slotset=slotset)
    current_tracker = []#manager.add_or_get_tracker(tracker_id)
    current_tracker.add_event(event)
    #db_manager.insert_event_mongo(event.event2dict())

    events = [event.event2dict(tracker_id) for event in current_tracker.events]
    response = dict()
    response['id'] = tracker_id
    response['events'] = events
    return json(response)

@app.route("/user/send",methods=['GET','POST','OPTIONS'])
async def user_query(request):
    print(request.args)
    print(request.form)
    query = request.args.get('query')
    if  query:
        if query == 'clear':
            agent.trackers.add_or_get_tracker('test_bot').clear_event()
            return text('all events clear')
        reply = agent.process(query)
        return text(reply)
    return text('I dont know how to answer it')



@app.route("/events/push",methods=['POST','OPTIONS'])
async def add_events(request):
    tracker_id = request.json.get('id')
    events = request.json.get('events')
    es = []
    for event in events:
        e = Event.dict2event(event)
        db_manager.insert_event_mongo(event)
        es.append(e)
    current_tracker = manager.add_or_get_tracker(tracker_id)
    current_tracker.add_muti_events(es)
    events = [event.event2dict(tracker_id) for event in current_tracker.events]
    response = dict()
    response['id'] = tracker_id
    response['events'] = events
    return json(response)

@app.route("/events/clear",methods=['POST'])
async def clear_events(request):
    tracker_id = request.json.get('id')
    response = dict()
    if tracker_id:
        current_tracker = manager.add_or_get_tracker(tracker_id)
        current_tracker.clear_event()
        db_manager.clear_events(tracker_id)
        response['state'] = 'success'
    else:
        response['state'] = 'undefined_id'
    return json(response)

@app.route("/events/view",methods=['POST','OPTIONS'])
async def view_events(request):
    tracker_id = request.json.get('id')
    response = dict()
    if tracker_id:
        current_tracker = manager.add_or_get_tracker(tracker_id)
        #current_tracker.view_event()
        events = [event.event2dict() for event in current_tracker.events]
        response = dict()
        response['id'] = tracker_id
        response['events'] = events
        response['state'] = 'success'
    else:
        response['state'] = 'undefined_id'
    return json(response)




@app.route('/text', methods=["GET","POST", "OPTIONS"])
def handle_request(request):
    return text('Hello world!')
#app.add_route(user_query,'user/send', methods=["GET","POST", "OPTIONS"])
#http://localhost:8000/
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

