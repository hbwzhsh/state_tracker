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
        elif self.action_name == 'action_find_place' :
            return self.find_poi()
        elif self.action_name == 'action_report_distance' :
            return self.find_poi()
        elif self.action_name == 'action_report_address' :
            return self.find_poi()
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

    def get_poi_data(self):
        return [
            {"address": "593 Arrowhead Way",
             "poi": "Chef Chu's",
             "type": "chinese restaurant"},
            {"address": "394 Van Ness Ave", "poi": "Coupa", "type": "coffee or tea place"},
            {"address": "408 University Ave",
             "poi": "Trader Joes",
             "type": "grocery store"},
            {"address": "113 Anton Ct", "poi": "Round Table", "type": "pizza restaurant"},
            {"address": "1313 Chester Ave",
             "poi": "Hacienda Market",
             "type": "grocery store"},
            {"address": "465 Arcadia Pl", "poi": "Four Seasons", "type": "rest stop"},
            {"address": "830 Almanor Ln", "poi": "tai pan", "type": "chinese restaurant"},
            {"address": "773 Alger Dr",
             "poi": "Stanford Shopping Center",
             "type": "shopping center"},
            {"address": "53 University Av", "poi": "Shell", "type": "gas station"},
            {"address": "657 Ames Ave", "poi": "The Clement Hotel", "type": "rest stop"},
            {"address": "792 Bedoin Street",
             "poi": "Starbucks",
             "type": "coffee or tea place"},
            {"address": "329 El Camino Real", "poi": "The Westin", "type": "rest stop"},
            {"address": "383 University Ave",
             "poi": "Town and Country",
             "type": "shopping center"},
            {"address": "452 Arcadia Pl", "poi": "Safeway", "type": "grocery store"},
            {"address": "171 Oak Rd", "poi": "Topanga Mall", "type": "shopping center"},
            {"address": "842 Arrowhead Way",
             "poi": "Panda Express",
             "type": "chinese restaurant"},
            {"address": "704 El Camino Real",
             "poi": "Pizza Hut",
             "type": "pizza restaurant"},
            {"address": "899 Ames Ct",
             "poi": "Stanford Childrens Health",
             "type": "hospital"},
            {"address": "5677 southwest 4th street",
             "poi": "5677 southwest 4th street",
             "type": "certain address"},
            {"address": "5672 barringer street",
             "poi": "5672 barringer street",
             "type": "certain address"},
            {"address": "214 El Camino Real",
             "poi": "Stanford Express Care",
             "type": "hospital"},
            {"address": "638 Amherst St",
             "poi": "Sigona Farmers Market",
             "type": "grocery store"},
            {"address": "611 Ames Ave",
             "poi": "Palo Alto Medical Foundation",
             "type": "hospital"},
            {"address": "434 Arastradero Rd",
             "poi": "Ravenswood Shopping Center",
             "type": "shopping center"},
            {"address": "338 Alester Ave",
             "poi": "Midtown Shopping Center",
             "type": "shopping center"},
            {"address": "271 Springer Street",
             "poi": "Mandarin Roots",
             "type": "chinese restaurant"},
            {"address": "753 University Ave", "poi": "Comfort Inn", "type": "rest stop"},
            {"address": "669 El Camino Real",
             "poi": "P.F. Changs",
             "type": "chinese restaurant"},
            {"address": "915 Arbol Dr", "poi": "Pizza Chicago", "type": "pizza restaurant"},
            {"address": "333 Arbol Dr", "poi": "Travelers Lodge", "type": "rest stop"},
            {"address": "436 Alger Dr",
             "poi": "Palo Alto Cafe",
             "type": "coffee or tea place"},
            {"address": "5677 springer street",
             "poi": "5677 springer street",
             "type": "certain address"},
            {"address": "113 Arbol Dr", "poi": "Jing Jing", "type": "chinese restaurant"},
            {"address": "409 Bollard St", "poi": "Willows Market", "type": "grocery store"},
            {"address": "776 Arastradero Rd", "poi": "Dominos", "type": "pizza restaurant"},
            {"address": "269 Alger Dr",
             "poi": "Cafe Venetia",
             "type": "coffee or tea place"},
            {"address": "110 Arastradero Rd",
             "poi": "Papa Johns",
             "type": "pizza restaurant"},
            {"address": "550 Alester Ave", "poi": "Dish Parking", "type": "parking garage"},
            {"address": "578 Arbol Dr", "poi": "Hotel Keen", "type": "rest stop"},
            {"address": "9981 Archuleta Ave",
             "poi": "Peets Coffee",
             "type": "coffee or tea place"},
            {"address": "200 Alester Ave", "poi": "Valero", "type": "gas station"},
            {"address": "819 Alma St", "poi": "Whole Foods", "type": "grocery store"},
            {"address": "91 El Camino Real", "poi": "76", "type": "gas station"},
            {"address": "583 Alester Ave", "poi": "Philz", "type": "coffee or tea place"},
            {"address": "270 Altaire Walk",
             "poi": "Civic Center Garage",
             "type": "parking garage"},
            {"address": "610 Amarillo Ave",
             "poi": "Stanford Oval Parking",
             "type": "parking garage"},
            {"address": "347 Alta Mesa Ave", "poi": "jills house", "type": "friends house"},
            {"address": "880 Ames Ct", "poi": "Webster Garage", "type": "parking garage"},
            {"address": "864 Almanor Ln", "poi": "jacks house", "type": "friends house"},
            {"address": "56 cadwell street",
             "poi": "home_2",
             "type": "home"},
            {"address": "5671 barringer street",
             "poi": "home_3",
             "type": "home"},
            {"address": "528 Anton Ct",
             "poi": "Pizza My Heart",
             "type": "pizza restaurant"},
            {"address": "10 ames street",
             "poi": "home_1",
             "type": "home"},
            {"address": "580 Van Ness Ave", "poi": "toms house", "type": "friends house"},
            {"address": "481 Amaranta Ave",
             "poi": "Palo Alto Garage R",
             "type": "parking garage"},
            {"address": "783 Arcadia Pl", "poi": "Chevron", "type": "gas station"},
            {"address": "145 Amherst St", "poi": "Teavana", "type": "coffee or tea place"}
        ]

    def get_all_poi_type(self):
        return [u.get('type') for u in self.get_poi_data()]

    def get_all_poi_name(self):
        return [u.get('poi') for u in self.get_poi_data()]

    def get_pois(self,key,value):
        l = [u for u in self.get_poi_data() if value.lower() in u.get(key).lower()]
        return l

    def find_poi_by_name(self,name):
        l = self.get_pois('poi',name)
        if len(l)>0:
            return l[0]
        return None

    def find_poi_by_type(self,type):
        l = self.get_pois('type',type)
        if len(l)>0:
            return l[0]
        return None

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
        rsp = "sry,I dont know how to confirm your statement."
        if (not location) or (not date) or (not weather_attribute):
            if not location:

                rsp = 'Which city are you interested in? and what day?'+'[!lack info {}]'.format(','.join(lacks))
            elif not date:
                rsp = "What day are you interested in?"
        if date:
            true_weather = None
            for w in weathers:
                if w.get('location')and  w.get('location').lower() == location.get_entity_value().lower():
                    true_weather = w.get(date.get_entity_value())
            if  true_weather:
                if weather_attribute.get_entity_type().lower() in true_weather.lower():
                    rsp = 'Yes ,it will be '
                else:
                    rsp = 'No,it will be {} in {} on {}'.format(true_weather,location.get_entity_value(),date.get_entity_value())
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
                    true_weather = w.get(date.get_entity_value())
            if true_weather:
                rsp = 'The forecast show that it will be {} in {}'.format(true_weather,location.get_entity_value())
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
        if slot_d['event'] and (not slot_d['date'] or not slot_d['time'] or not slot_d['room']):
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

    def find_poi(self):
        slot_d = dict()
        for slotname in ['poi_type','poi_name']:
            slot = self.find_slot_by_type(slotname)
            if slot:
                slot_d[slotname] = slot.get_entity_value()
            else:
                slot_d[slotname] = None
        lacks = [k for k in slot_d.keys() if slot_d[k]==None]
        haves = [k for k in slot_d.keys() if slot_d[k]!=None]
        if 'poi_type' in haves and 'poi_name' not in haves:
            type = slot_d['poi_type']
            poi = self.find_poi_by_type(type)
            if poi:
                tmp = 'There is a {} around called {}'.format(type,poi['poi'])
                tmp += 'It is 5 miles away and located in {} .'.format(poi['address'])
            else:
                tmp = 'This is no {} around.sorry.'.format(type)
            return tmp
        elif 'poi_name' in haves and 'poi_type' not in haves:
            name = slot_d['poi']
            poi = self.find_poi_by_name(name)
            if poi:

                tmp = '{} is 5 miles away and located in {} .'.format(name ,poi['address'])
            else:
                tmp = 'I cant find the poi {}.'.format(name)
            return tmp







