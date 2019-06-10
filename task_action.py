from entity import Entity
from action import Action
import json
import random
import requests
import reader
import sample_data
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

        action_utterance_dict = reader.read_temp("action_tmp.txt")
        return action_utterance_dict.get(action)

    def get_weather_data(self):
        return sample_data.sample_weather_data()
    def get_schedule_data(self):
        return sample_data.sample_schedule_data()

    def get_poi_data(self):
        return sample_data.sample_poi_data()

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

    def get_city_weather(self,city):
        url  ="http://api.openweathermap.org/data/2.5/weather?q={}&APPID=0d11a89593c5bde6a3a7cf1094eac1b1".format(city)
        r = requests.get(url)
        return r.json()

    def get_city_weather_forecast(self,city):
        url  ="http://api.openweathermap.org/data/2.5/forecast?q={}&APPID=0d11a89593c5bde6a3a7cf1094eac1b1&units=metric".format(city)
        r = requests.get(url)
        return r.json()


    def get_weathers_from_json(self,rsp):
        weathers = rsp.get("list")
        new_d = []
        for weather in weathers:
            d = {}

            tmp = weather.get("main")
            degree= 20
            if tmp:
                degree = tmp.get("temp")
            d["temperture"] = degree
            dpt = ""
            if weather.get("weather"):
                dpt = weather.get("weather")[0].get("description")
            d["description"] = dpt
            dt_txt = weather.get("dt_txt")
            #"2019-06-05 18:00:00"
            date_seg = dt_txt.split(" ")[0]
            time_seg = dt_txt.split(" ")[1]
            year,month,day = [int(u) for u in date_seg.split("-")]
            hour,min,sec = [int(u) for u in time_seg.split(":")]

            d["year"] = year
            d["month"] = month
            d["day"] = day
            d["hour"] = hour
            new_d.append(d)
            # import time
            # current = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()
        return new_d


    def get_many_days_weather(self,date_str,today_date,weathers_d):
        target_dates = []
        n = 2
        maps = {"two": 2, "three": 3, "four": 4, "five": 5}
        for time, nums in maps.items():
            if time in date_str:
                n = nums
        for day in range(n):
            target_dates.append(today_date + day)
        true_weathers = {}
        for w in weathers_d:
            for i, target_date in enumerate(target_dates):
                if w.get("day") == target_date:
                    true_weather = w.get("description")
                    true_weathers[target_date] = true_weather
        return target_dates,true_weathers

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
                rsp = 'Which city are you interested in? or what day?'+'[!lack info {}]'.format(','.join(lacks))
                return(rsp)
            elif not date:
                rsp = "What day are you interested in?"
                return(rsp)
        if date:
            true_weather = None
            # for w in weathers:
            #     if w.get('location')and  w.get('location').lower() == location.get_entity_value().lower():
            #         true_weather = w.get(date.get_entity_value())

            ask_many_days = "next" in date.get_entity_value() and "day" in date.get_entity_value()
            if date.get_entity_value() in ["today","tomorrow"]or ask_many_days:
                if not location:
                    raise ValueError("not location found")
                forecast = self.get_city_weather_forecast(location.get_entity_value())
                weathers_d = self.get_weathers_from_json(forecast)
                import time
                today_date = time.strftime("%d", time.localtime())
                today_date = int(today_date)
                target_date = today_date
                if date.get_entity_value() == "tomorrow":
                    target_date = today_date+1
                if ask_many_days:
                    all_true = True
                    date_str = date.get_entity_value()
                    target_dates,true_weathers = self.get_many_days_weather(self, date_str, today_date, weathers_d)
                    for i, target_date in enumerate(target_dates):
                        true_weather = true_weathers[target_date]
                        if weather_attribute.get_entity_value() in true_weather:
                            all_true = all_true and True
                        else:
                            all_true = all_true and False

                    if all_true:
                        rsp = "Yes,it will be "
                    else:
                        rsp = "No,it will be "
                    for i, target_date in enumerate(target_dates):
                        true_weather = true_weathers[target_date]
                        rsp += "{} on {}th".format(true_weather, target_date)
                        if i == len(target_dates) - 1:
                            rsp += "."
                        else:
                            rsp += ","


                    return rsp


                for w in weathers_d:
                    if w.get("day") == target_date:
                        true_weather = w.get("description")
            if  true_weather:
                if weather_attribute.get_entity_value().lower() in true_weather.lower():
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
            #weathers = self.get_weather_data()
            ask_many_days = "next" in date.get_entity_value() and "day" in date.get_entity_value()
            if date.get_entity_value() in ["today","tomorrow"]or ask_many_days:
                if not location:
                    raise ValueError("not location found")
                forecast = self.get_city_weather_forecast(location.get_entity_value())
                weathers_d = self.get_weathers_from_json(forecast)
                import time
                today_date = time.strftime("%d", time.localtime())
                today_date = int(today_date)
                target_date = today_date
                if date.get_entity_value() == "tomorrow":
                    target_date = today_date+1
                    for w in weathers_d:
                        if w.get("day") == target_date:
                            true_weather = w.get("description")
                elif ask_many_days:
                    date_str = date.get_entity_value()
                    target_dates,true_weathers = self.get_many_days_weather(self, date_str, today_date, weathers_d)
                    rsp = "It will be "
                    for i, target_date in enumerate(target_dates):
                        true_weather = true_weathers[target_date]
                        rsp += "{} on {}th".format(true_weather, target_date)
                        if i == len(target_dates) - 1:
                            rsp += "."
                        else:
                            rsp += ","
                    return rsp

            # for w in weathers:
            #     if w.get('location') and w.get('location').lower() == location.get_entity_value().lower():
            #         true_weather = w.get(date.get_entity_value())
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

#
# print(Navigate("",{}).get_city_weather("London"))
#
        # action_utterance_dict =  {
        # 'action_goodbye': 'You are welcome,have a wonderful day.',
        # 'action_morehelp': 'Anything else I can help you with today?',
        # 'action_ok': 'No problem.',
        # 'action_report_distance': 'Mandarin Roots is [found_distance] away.',
        # 'action_report_address': 'The address is [found_address] and there is moderate traffic on the way to the location.',
        # 'action_affirm_want_direction': ' Would you like directions?',
        # 'action_report_traffic': 'The current route  have [found_traffic] traffic',
        # 'action_set_navigation': 'I\'ll send the route with no traffic on your screen, drive carefully!',
        # 'action_ask_location': 'Which city would you like the weather forecast for [inform_date]?',
        # 'action_check_weather': 'The weather in [inform_location] will be snowy on Monday and then on Sunday, cloudy on Tuesday, raining on Wednesday, blizzard on Thursday, lowest temperature on Friday, dry on Saturday',
        # 'action_clearify': 'The forecast does show show that it will be windy in Manhattan today or tomorrow.',
        # 'action_report_event': 'Your next football activity is going to be on Thursday at 11 am with your aunt.',
        # 'action_set_reminder': 'Setting a reminder for your conference next Wednesday at 7 pm with HR.',
        # 'action_ask_newevent_time': ' What is the date & time?'
        # }


