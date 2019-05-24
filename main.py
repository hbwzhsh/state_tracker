from sanic import Sanic
from sanic.response import json,text
from entity import Entity
from TrackerManager import TrackerManager
from slot_tool import SlotTool
from event import Event
app = Sanic()
manager = TrackerManager()

@app.route("/")
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

@app.route("/event/add",methods=['POST'])
async def add_event(request):
    tracker_id = request.json.get('id')
    slot_d = request.json.get('slots')
    type = request.json.get('type')
    text = request.json.get('text')
    if slot_d:
        slotset = SlotTool.dict2slotset(slot_d)
    else:
        slotset = []
    event = Event(type,text,slotset=slotset)
    current_tracker = manager.add_or_get_tracker(tracker_id)
    current_tracker.add_event(event)
    events = [event.event2dict() for event in current_tracker.events]
    response = dict()
    response['id'] = tracker_id
    response['events'] = events
    return json(response)

@app.route("/events/push",methods=['POST'])
async def add_events(request):
    tracker_id = request.json.get('id')
    events = request.json.get('events')
    es = []
    for event in events:
        e = Event.dict2event(event)
        es.append(e)
    current_tracker = manager.add_or_get_tracker(tracker_id)
    current_tracker.add_muti_events(es)
    events = [event.event2dict() for event in current_tracker.events]
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
        response['state'] = 'success'
    else:
        response['state'] = 'undefined_id'
    return json(response)

@app.route("/inform/add",methods=['POST'])
async def great(request):
    slot_d = request.json.get('slots')
    tracker_id = request.json.get('id')
    current_slots =  SlotTool.dict2slotset(slot_d)#get_sample_slot()
    for s in current_slots:
        s.print_entity()
    if not tracker_id:
        tracker_id ='test_bot'
    current_tracker = manager.add_or_get_tracker(tracker_id)
    inform_slots = [s.inform_slot() for s in current_slots]
    previous_slots = current_tracker.slotset
    updated_slots = current_tracker.update(previous_slots, inform_slots)
    current_tracker.slotset = updated_slots
    # for s in updated_slots:
    #     s.print_entity()
    # SlotTool.dict2slotset({'food': 'chinese'})
    response = dict()
    response['id'] = tracker_id
    response['slots'] = SlotTool.slotset2dict(current_tracker.slotset)
    return json(response)




@app.route('/text')
def handle_request(request):
    return text('Hello world!')

#http://localhost:8000/
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

