from entity import Entity
from action import Action
import json
import random
data_path = 'C://CamRestDB.json'
class Task():
    def __init__(self,name):
        self.action_name = name
        self.all_slot = ['pricerange','food','area']
        self.inform_slot = []
        self.slots = []

    def run(self,slots,required):
        slots = [slot for slot in slots if slot.get_entity_type() in required]
        self.inform_slot = [slot.get_entity_type() for slot in slots]
        matches = self.search(self.load_data(),slots)
        s,rs = self.show_rests(matches)
        return s,rs

    # get the slotname with prefix like request_,
    def search(self,dicts,slots):
        #slots = [s for s in slots]
        for s in slots:
            new = []
            for d in dicts:
                v = d.get(s.get_entity_type().split('_')[-1])
                if v :
                    if v.lower() == s.get_entity_value().lower():
                        new.append(d)
            dicts = new
            #dicts = [d for d in dicts if d.get(s.get_entity_type().split('_')[-1]).lower() == s.get_entity_value().lower()]
        return dicts

    def load_data(self):
        file = open(data_path)
        rests = json.load(file)
        file.close()
        return rests

    def show_rests(self,rests):
        if len(rests) == 0:
            return 'No matches found,please change your preference',None
        if len(rests)==1:
            s = 'There is one restaurant that meets that criteria,{}.Do you want any infomation?'.format(rests[0].get('name'))+'\n'
        elif len(rests) <5 :
            s = 'There are {} resturants found.We recommend {} to you.Do you want any infomation?'.format(len(rests),rests[0].get('name')) + '\n'
        else:

            have_inform = self.inform_slot
            print(have_inform)
            have_inform = [s[7:] for s in have_inform if s.startswith('inform_')]
            request = ['food','pricerange','area']
            no_inform = [s for s in request if s not in have_inform]
            n = random.randint(0,len(no_inform)-1)
            new_inform = no_inform[n]
            tmp  ='. What is your other preference(food,pricerange,area)?'
            if new_inform == 'food':
                tmp = 'What kind of food do you want?'
            elif new_inform == 'area':
                tmp = 'Which area do you want the restaurants to be located in ?'
            elif new_inform == 'pricerange':
                tmp = 'Do you want cheap  pricerange or expensive pricerange?'

            s = 'There are {} resturants found .'.format(len(rests))+ tmp
        # for r in rests:
        #     name = r.get('name')
        #     food = r.get('food')
        #     area = r.get('area')
        #     s += '{} serves {} food and is located in the {} of the town'.format(name,food,area)
        #     s += '\n'
        return s,rests

# slot_food = Entity('food')
# slot_food.set_value('chinese')
# slot_area = Entity('area')
# slot_area.set_value('south')
# task = Task('action_search_rest')
# task.run([slot_food,slot_area],[slot_food.type_name,slot_area.type_name])
#print(task.load_rest()[0])

class Navigate(Action):
    def run(self):
        if self.action_name == 'action_check_weather':
            return self.how_weather_q()
        elif self.action_name == 'action_clearify':
            return self.is_weather_q()
        elif self.action_name == 'action_set_reminder':
            return self.set_reminder()
        elif self.action_name == 'action_report_event':
            return self.search_reminder()
        else:
            new_slots = []
            for s in self.slots:
                if s.type_name.startswith('inform'):
                    continue
                slots= Entity('inform_'+s.type_name)
                slots.set_value(s.get_entity_value())
                new_slots.append(slots)

            return self.get_filled_response(new_slots)

    def set_name_slot(self,name,slots):
        self.action_name = name
        self.slots = slots

    def get_action_temp(self,action):
        action_utterance_dict =  {
        'action_goodbye': 'You are welcome,have a wonderful day.',
        'action_morehelp': 'Anything else I can help you with today?',
        'action_ok': 'No problem.',
        'action_report_distance': 'Mandarin Roots is [found_distance] away.',
        'action_report_address': 'The address is [found_address] and there is moderate traffic on the way to the location.',
        'action_affirm_want_direction': ' Would you like directions?',
        'action_report_traffic': 'The current route  have [found_traffic] traffic',
        'action_set_navigation': 'I\'ll send the route with no traffic on your screen, drive carefully!',
        'action_ask_location': 'Which city would you like the weather forecast for [inform_date]?',
        'action_check_weather': 'The weather in [inform_location] will be snowy on Monday and then on Sunday, cloudy on Tuesday, raining on Wednesday, blizzard on Thursday, lowest temperature on Friday, dry on Saturday',
        'action_clearify': 'The forecast does show show that it will be windy in Manhattan today or tomorrow.',
        'action_report_event': 'Your next football activity is going to be on Thursday at 11 am with your aunt.',
        'action_set_reminder': 'Setting a reminder for your conference next Wednesday at 7 pm with HR.',
        'action_ask_newevent_time': ' What is the date & time?'
        }
        return action_utterance_dict.get(action)

    def get_weather_data(self):
        return [
          {
            "monday": "frost, low of 50F, high of 60F",
            "tuesday": "rain, low of 20F, high of 30F",
            "friday": "hot, low of 70F, high of 80F",
            "wednesday": "snow, low of 80F, high of 100F",
            "thursday": "cloudy, low of 60F, high of 80F",
            "sunday": "cloudy, low of 50F, high of 70F",
            "location": "carson",
            "saturday": "snow, low of 20F, high of 30F",
            "today": "monday"
          },
          {
            "monday": "dry, low of 20F, high of 30F",
            "tuesday": "clear skies, low of 90F, high of 100F",
            "friday": "cloudy, low of 30F, high of 50F",
            "wednesday": "drizzle, low of 40F, high of 50F",
            "thursday": "overcast, low of 60F, high of 70F",
            "sunday": "dry, low of 70F, high of 90F",
            "location": "new york",
            "saturday": "dry, low of 40F, high of 50F",
            "today": "monday"
          },
          {
            "monday": "foggy, low of 70F, high of 90F",
            "tuesday": "hot, low of 70F, high of 90F",
            "friday": "blizzard, low of 20F, high of 40F",
            "wednesday": "snow, low of 90F, high of 100F",
            "thursday": "windy, low of 60F, high of 80F",
            "sunday": "raining, low of 90F, high of 100F",
            "location": "brentwood",
            "saturday": "raining, low of 60F, high of 80F",
            "today": "monday"
          },
          {
            "monday": "foggy, low of 20F, high of 40F",
            "tuesday": "stormy, low of 60F, high of 70F",
            "friday": "cloudy, low of 40F, high of 50F",
            "wednesday": "overcast, low of 60F, high of 80F",
            "thursday": "rain, low of 80F, high of 100F",
            "sunday": "warm, low of 70F, high of 80F",
            "location": "san mateo",
            "saturday": "dry, low of 70F, high of 80F",
            "today": "monday"
          },
          {
            "monday": "rain, low of 80F, high of 90F",
            "tuesday": "rain, low of 50F, high of 60F",
            "friday": "warm, low of 70F, high of 80F",
            "wednesday": "foggy, low of 20F, high of 40F",
            "thursday": "dry, low of 30F, high of 40F",
            "sunday": "snow, low of 30F, high of 40F",
            "location": "oakland",
            "saturday": "cloudy, low of 70F, high of 80F",
            "today": "monday"
          },
          {
            "monday": "windy, low of 70F, high of 80F",
            "tuesday": "cloudy, low of 60F, high of 70F",
            "friday": "rain, low of 70F, high of 90F",
            "wednesday": "rain, low of 30F, high of 50F",
            "thursday": "clear skies, low of 60F, high of 80F",
            "sunday": "overcast, low of 30F, high of 40F",
            "location": "cleveland",
            "saturday": "raining, low of 40F, high of 60F",
            "today": "monday"
          },
          {
            "monday": "cloudy, low of 70F, high of 90F",
            "tuesday": "foggy, low of 20F, high of 30F",
            "friday": "windy, low of 30F, high of 40F",
            "wednesday": "clear skies, low of 90F, high of 100F",
            "thursday": "clear skies, low of 40F, high of 60F",
            "sunday": "stormy, low of 60F, high of 80F",
            "location": "grand rapids",
            "saturday": "rain, low of 90F, high of 100F",
            "today": "monday"
          }
        ]
    def get_schedule_data(self):
        return [
          {
            "room": "conference room 100",
            "agenda": "discuss the company picnic",
            "time": "7 pm",
            "date": "thursday",
            "party": "HR",
            "event": "meeting"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "2 pm",
            "date": "thursday",
            "party": "-",
            "event": "swimming activity"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "11 am",
            "date": "monday",
            "party": "-",
            "event": "football activity"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "7 pm",
            "date": "monday",
            "party": "-",
            "event": "yoga activity"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "7 pm",
            "date": "friday",
            "party": "-",
            "event": "dentist"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "10 am",
            "date": "wednesday",
            "party": "-",
            "event": "lab appointment"
          }
        ]+[
          {
            "room": "-",
            "agenda": "-",
            "time": "1 pm",
            "date": "wednesday",
            "party": "-",
            "event": "lab appointment"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "7 pm",
            "date": "monday",
            "party": "-",
            "event": "doctor appointment"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "11 am",
            "date": "thursday",
            "party": "-",
            "event": "dentist"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "2 pm",
            "date": "thursday",
            "party": "-",
            "event": "swimming activity"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "2 pm",
            "date": "wednesday",
            "party": "-",
            "event": "yoga activity"
          },
          {
            "room": "-",
            "agenda": "-",
            "time": "2 pm",
            "date": "tuesday",
            "party": "-",
            "event": "dentist appointment"
          }
        ]
    def find_slot_by_type(self,slot_type):
        for slot in self.slots:
            if slot.get_entity_type() == slot_type :
                return slot
            if slot.get_entity_type() == 'inform_'+slot_type:
                return slot.cut_inform()
        return None

    def is_weather_q(self):
        weathers = self.get_weather_data()
        for s in self.slots:
            if s.get_entity_type() == 'weekly_time':
                print(s.get_entity_value)

        location = self.find_slot_by_type('location')
        date = self.find_slot_by_type('date')
        weather_attribute = self.find_slot_by_type('weather_attribute')
        lacks = []
        if not location:
            lacks.append('location')
        if not date:
            lacks.append('date')
        if not weather_attribute:
            lacks.append('weather')

        if (not location) or (not date) or (not weather_attribute):
            rsp = 'Which city are you interested in? and what day?'+'[!lack info {}]'.format(','.join(lacks))
        true_weather = None
        for w in weathers:
            if w.get('location')and  w.get('location').lower() == location.get_entity_value().lower():
                true_weather = w.get(date)
        if weather_attribute.lower() in true_weather.lower():
            rsp = 'Yes ,it will be '
        else:
            rsp = 'No,it will be {} in {} on {}'.format(weather_attribute,location,date)
        return rsp
    #"what's the weather like today"
    def how_weather_q(self):
        location = self.find_slot_by_type('location')
        date = self.find_slot_by_type('date')
        weather_attribute = self.find_slot_by_type('weather_attribute')
        lacks = []
        if not location:
            lacks.append('location')
        if not date:
            lacks.append('date')
        if not weather_attribute:
            lacks.append('weather')
        if (not location) or (not date) :
            print('[!lack info {}]'.format(','.join(lacks)))
            rsp = 'Which city are you interested in? and what day?'
        else:
            true_weather = None
            weathers = self.get_weather_data()
            for w in weathers:
                if w.get('location') and w.get('location').lower() == location.get_entity_value().lower():
                    true_weather = w.get(date)
            if true_weather:
                rsp = 'The forecast show that it will be {} in {}'.format(true_weather,location)
            else:
                rsp = 'sry,cannot find a weather forecast'
        return rsp

    def set_reminder(self):
        slot_d = dict()
        for slotname in ['room','date','time','event']:
            slot = self.find_slot_by_type(slotname)
            if slot:
                slot_d[slotname] = slot.get_entity_value()
            else:
                slot_d[slotname] = None
        lacks = [k for k in slot_d.keys() if slot_d[k]==None]
        rsp = ''
        if 'date' in lacks or 'time' in lacks:
            print('! lack date and time')
            rsp = 'what date and time is the activity?'
        elif 'event' in lacks:
            print('! unknown event')
            rsp = 'What is the activity?'
        else:
            rsp = 'set a reminder for your {} on {} {}'.format(slot_d['event'],slot_d['date'],slot_d['time'])
        return rsp

    def search_reminder(self):
        slot_d = dict()
        rsp = ""
        for slotname in ['room','date','time','event']:
            slot = self.find_slot_by_type(slotname)
            if slot:
                slot_d[slotname] = slot.get_entity_value()
            else:
                slot_d[slotname] = None
        found = False
        if slot_d['event'] and (not slot_d['date'] or not slot_d['time']):
            for event_item in self.get_schedule_data():
                if event_item.get('event') == slot_d['event']:
                    rsp = 'The {} is on {} at {}'.format(event_item.get('event'),event_item.get('date'),event_item.get('time'))
                    found = True
        if (slot_d['date'] or slot_d['time'] ) and (not slot_d['event']):
            for event_item in self.get_schedule_data():
                if slot_d['date'] :
                    if event_item.get('date') == slot_d['date'] :
                        rsp = 'The {} is on {} at {}'.format(event_item.get('event'),event_item.get('date'),event_item.get('time'))
                        found = True
                if slot_d['time'] :
                    if  ' '.join(event_item.get('time').split(' ')) == slot_d['time']:
                        rsp = 'The {} is on {} at {}'.format(event_item.get('event'),event_item.get('date'),event_item.get('time'))
                        found = True
        if not found:
            rsp = 'sry ,you have not set your event'
        return rsp






