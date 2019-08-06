import random

import requests

from action_api import sample_data
from dialogue_state import reader
from dialogue_state.action import Action
from dialogue_state.entity import Entity
from loggers import dbot_online_logger
from model.slot_filling_model import HelperNER
from path_config import *
from pprint import pprint

data_path = 'C://CamRestDB.json'
weekdays  =["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
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
        '''

        :param dicts:
        :param slots:
        :return:
        '''
        for s in slots:
            new = []
            for d in dicts:
                v = d.get(s.get_entity_type().split('_')[-1])
                if v :
                    if v.lower() == s.get_entity_value().lower():
                        new.append(d)
            dicts = new
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
        return s,rests

class Navigate(Action):

    def __init__(self,name,d,reminder_manager,domain=None):
        self.action_name = name
        self.utterance = None
        self.utterance_dict = d
        self.template = None
        self.slot_values = None#get_slot_values_dict()
        self.slots = []
        self.reminders = reminder_manager
        self.domain  = domain
        self.inform_slot = []

    # get the slotname with prefix like request_,
    def search(self,dicts,slots):
        '''

        :param dicts:
        :param slots:
        :return:
        '''
        for s in slots:
            new = []
            for d in dicts:
                v = d.get(s.cut_inform().get_entity_type())
                if v :
                    if isinstance(s.get_entity_value(),str):
                        if v.lower() == s.get_entity_value().lower():
                            new.append(d)
            dicts = new
        return dicts

    def load_data(self):
        file = open(data_path)
        rests = json.load(file)
        file.close()
        return rests


    def run(self):
        '''
        if the action name is any prefex defined method
        run the corresponding method
        otherwise find and return the utter template of the utter name
        :return:text reply
        '''
        if self.action_name == 'action_check_weather':
            return self.how_weather_q()
        elif self.action_name == 'action_clearify':
            return self.is_weather_q()
        elif self.action_name == 'action_set_reminder':
            return self.set_reminder()
        elif self.action_name == 'action_report_event':
            return self.search_reminder()
        elif self.action_name == 'action_ask_newevent_time':
            return self.set_reminder()
        elif self.action_name == 'action_find_place' :
            return self.find_poi()
        elif self.action_name == 'action_report_distance' :
            return self.find_poi()
        elif self.action_name == 'action_report_address' :
            return self.find_poi()
        elif self.action_name == 'action_show_rest' :
            return self.search_rest(self.domain.slots,['area','food','pricerange'])
        elif self.action_name == 'action_answer_attribute':
            return self.answer_attribute()

        elif self.action_name == 'action_ask_next_slot':
            """
            take current unfilled slot name
            find the utterance template by the first unfilled slot name
            :rtype: str
            """
            lack_slotnames, d = self.domain.get_lack_slotname_and_dict()
            if lack_slotnames == []:
                return "all slots are filled"
            self.input_slot = lack_slotnames[0]
            list_utter = Navigate.get_tmp_by_action("utter_list_" + lack_slotnames[0])
            ask_utter = Navigate.get_tmp_by_action("utter_ask_" + lack_slotnames[0])
            if list_utter == None:
                list_utter = ""
            if ask_utter == None:
                ask_utter = ""
            return "\n".join([list_utter,ask_utter])
        else:
            new_slots = []
            slots = [s.cut_inform() for s in self.slots]
            for s in slots:
                if s.type_name.startswith('inform'):
                    continue
                slot_= Entity('inform_'+s.type_name)
                slot_.set_value(s.get_entity_value())
                new_slots.append(slot_)

            return self.get_filled_response(new_slots)

    def answer_attribute(self):
        user_q = self.domain.get_current_user_query()
        requests = ['number','address','postcode']
        request_to_answer = [u for u in requests if u.lower() in user_q.lower()]
        rsp = ""
        match = self.domain.get_kb_data("rest_matched")
        if match:
            match = match[0]
        for req in request_to_answer:
            match
            tmp = Navigate.get_tmp_by_action("utter_answer_" + req)
            value = match.get(req)
            if not value:
                value = "unknown"
            tmp = tmp.format(match["name"],value)
            rsp += tmp
        return rsp

    def set_name_slot(self,name,slots):
        self.action_name = name
        self.slots = slots

    def search_rest(self,slots,required):
        slots = [slot for slot in slots if slot.get_entity_type() in required]
        self.inform_slot = [slot.get_entity_type() for slot in slots]
        matches = self.search(self.load_data(),slots)
        self.domain.set_kb_data('rest_matched',matches)
        s = self.show_rests(matches)
        return s

    def show_rests(self,rests):
        if len(rests) == 0:
            return 'No matches found,please change your preference'
        if len(rests)==1:
            s = 'There is one restaurant that meets that criteria,{}.Do you want any infomation?'.format(rests[0].get('name'))+'\n'
        elif len(rests) <5 :
            s = 'There are {} resturants found.We recommend {} to you.Do you want any infomation?'.format(len(rests),rests[0].get('name')) + '\n'
        else:
            s = 'There are {} resturants found .'.format(len(rests))
        return s

    def get_action_temp(self,action):
        return Navigate.get_tmp_by_action(action)

    @staticmethod
    def get_tmp_by_action(action):
        if isinstance(action,str)==False:
            raise TypeError("str expected but {} is {}".format(action,type(action)))
        action_utterance_dict = reader.read_temp(action_tmp_path)
        return action_utterance_dict.get(action)


    def get_weather_data(self):
        return sample_data.sample_weather_data()

    def get_schedule_data(self):
        return sample_data.sample_schedule_data() + self.reminders.events

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


    def get_many_days_weather(self,date_str,weathers_d):
        target_dates = []
        n = 2
        maps = {"two": 2, "three": 3, "four": 4, "five": 5}
        for time, nums in maps.items():
            if time in date_str:
                n = nums
        for day in range(n):
            target_dates.append(HelperNER.next_date(n))
        true_weathers = {}
        for w in weathers_d:
            for i, full_date in enumerate(target_dates):
                dbot_online_logger.debug(full_date)
                year,month,day = full_date.split('-')
                if w.get("day") == day and w.get("month") == month:
                    true_weather = w.get("description")
                    true_weathers[full_date] = true_weather
        return target_dates,true_weathers

    def is_weather_q(self):
        '''
        will it rain in new york?
        :return:
        '''
        for s in self.slots:
            if s.get_entity_type() == 'weekly_time':
                print(s.get_entity_value)
            if s.get_entity_type() == 'inform_question_word':
                if s.get_entity_value().lower() in ["what weather"]:
                    print("convert to check")
                    return self.how_weather_q()

        #weathers = self.get_weather_data()
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
            date_string = date.get_entity_value()
            ask_many_days = "next" in date.get_entity_value() and "day" in date.get_entity_value()
            if date.get_entity_value() in ["today","tomorrow"]or ask_many_days or date_string in weekdays:
                if not location:
                    raise ValueError("not location found")
                forecast = self.get_city_weather_forecast(location.get_entity_value())
                weathers_d = self.get_weathers_from_json(forecast)
                import time
                today_date = time.strftime("%d", time.localtime())
                today_date = int(today_date)
                target_date = today_date
                if date.get_entity_value() == "tomorrow":
                    target_date =  HelperNER.date2day(HelperNER.tomorrow())
                    target_month =  HelperNER.date2month(HelperNER.tomorrow())
                    weather = self.get_weather_by_month_day(weathers_d,target_month,target_date)
                    if weather:
                        true_weather = weather.get("description")
                elif   date.get_entity_value()  in weekdays:
                    target_month,target_date = HelperNER().weekday2date(date.get_entity_value())
                    weather = self.get_weather_by_month_day(weathers_d,target_month,target_date)
                    if weather:
                        true_weather = weather.get("description")
                elif ask_many_days:
                    all_true = True
                    date_str = date.get_entity_value()
                    target_dates,true_weathers = self.get_many_days_weather(self, date_str, weathers_d)
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
                        rsp += "{} on {}".format(true_weather, target_date)
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

    def get_weather_by_day(self,weathers_d,target_month = None,target_date=None):
        for w in weathers_d:
            if w.get("day") == target_date and w.get("month") == target_month:
                return w
        return None

    def get_weather_by_month_day(self,weathers_d,target_month = None,target_date=None):
        for w in weathers_d:
            if w.get("day") == target_date and w.get("month") == target_month:
                return w
        return None

    def how_weather_q(self):
        '''
        handle the query like :what is the weather like today
        :return:
        '''

        dbot_online_logger.debug("how weather q ")
        for s in self.slots:
            if s.get_entity_type() == 'weekly_time':
                print(s.get_entity_value)
            if s.get_entity_type() == 'inform_question_word':
                if s.get_entity_value().lower() in ["is","will","are","whether","if","if weather"]:
                    print("convert to clearify")
                    return self.is_weather_q()
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
            rsp = ""
            if not location:
                dbot_online_logger.debug(" (not location) ")
                rsp += 'Which city are you interested in?'
            if not date:
                dbot_online_logger.debug("(not date) ")
                rsp += 'what is the date that  you  are interested in?'
            dbot_online_logger.debug('[!lack info {}]'.format(','.join(lacks)))

        else:
            true_weather = None
            #weathers = self.get_weather_data()
            date_string = date.get_entity_value()
            ask_many_days =("next" in date_string)  and ("day" in date_string)
            if date_string in ["today","tomorrow"]or ask_many_days or date_string in weekdays:
                if not location:
                    raise ValueError("not location found")
                forecast = self.get_city_weather_forecast(location.get_entity_value())
                dbot_online_logger.debug("forecast")
                dbot_online_logger.debug(forecast)
                weathers_d = self.get_weathers_from_json(forecast)
                dbot_online_logger.debug("weathers_d")
                dbot_online_logger.debug(weathers_d)
                import time
                today_date = time.strftime("%d", time.localtime())
                today_date = int(today_date)
                target_date = today_date
                if date.get_entity_value() in ["today","tomorrow"]:
                    dbot_online_logger.debug("ask today  | tomorrow")
                    if date.get_entity_value() == 'tomorrow':
                        dbot_online_logger.debug("tomorrow")
                        target_date = HelperNER.date2day(HelperNER.tomorrow())
                        target_month = HelperNER.date2month(HelperNER.tomorrow())
                    elif date.get_entity_value() == 'today':
                        dbot_online_logger.debug("today")
                        target_date = HelperNER.date2day(HelperNER.get_current_date(True))
                        target_month = HelperNER.date2month(HelperNER.get_current_date(True))
                    weather = self.get_weather_by_month_day(weathers_d, target_month, target_date)
                    if weather:
                        true_weather = weather.get("description")
                elif date.get_entity_value()  in weekdays:
                    dbot_online_logger.debug("how weather q, ask about weekdays")
                    target_month,target_date = HelperNER().weekday2date(date.get_entity_value())
                    weather = self.get_weather_by_month_day(weathers_d,target_month,target_date)
                    dbot_online_logger.debug(weather)
                    if weather:
                        true_weather = weather.get("description")
                elif ask_many_days:
                    dbot_online_logger.debug("how weather q, ask_many_days")
                    date_str = date.get_entity_value()
                    target_dates,true_weathers = self.get_many_days_weather(self, date_str,  weathers_d)
                    rsp = "It will be "
                    for i, target_date in enumerate(target_dates):
                        true_weather = true_weathers[target_date]
                        rsp += "{} on {}th".format(true_weather, target_date)
                        if i == len(target_dates) - 1:
                            rsp += "."
                        else:
                            rsp += ","
                    return rsp

            if true_weather:
                dbot_online_logger.debug("---------true weather is found")
                rsp = 'The forecast show that it will be {} in {} on {}'.format(true_weather,location.get_entity_value(),date_string)
            else:
                dbot_online_logger.debug("----------no true weather is found")
                rsp = 'sry,cannot find a weather forecast'
        return rsp

    def set_reminder(self):
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        slot_d = dict()
        date  = slot_d.get('date')
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
            rsp = 'What time shall I set the reminder?'
        elif 'event' in lacks:
            print('! unknown event')
            rsp = 'What is the activity?'
        else:
            rsp = 'set a reminder for your {} on {} {}'.format(slot_d['event'],slot_d['date'],slot_d['time'])
            date_value = slot_d['date'].lower()
            date = self.date_string(date_value)
            self.reminders.add_new_event(eventname=slot_d['event'],time=slot_d['time'],date=date)
        return rsp


    def date_string(self,date_value):
        date = date_value
        if date_value in ['today', 'tomorrow', 'the day after tomorrow']:
            if date_value == 'today':
                date = HelperNER.get_current_date(True)
            elif date_value == 'tomorrow':
                date = HelperNER.tomorrow()
            elif date_value == 'the day after tomorrow':
                date = HelperNER.day_after_tomorrow()
            dbot_online_logger.debug("the date is converted into " + date)
        elif date_value in weekdays:
            date = HelperNER.weekday2date(date_value)
            dbot_online_logger.debug("the date is converted into (weekday) " + date)
        return date

    def rsp_search_event_time(self,event_item):
        return 'The {} is on {} at {}'.format(event_item.get('event'),event_item.get('date'),event_item.get('time'))

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
        # event known, but date ,time or room is unknown
        if slot_d['event'] and (not slot_d['date'] or not slot_d['time'] or not slot_d['room']):
            for event_item in self.get_schedule_data():
                if event_item.get('event') == slot_d['event']:
                    rsp = self.rsp_search_event_time(event_item)
                    found = True
        # event unknown but time
        if (slot_d['date'] or slot_d['time'] ) and (not slot_d['event']):
            for event_item in self.get_schedule_data():
                if slot_d['date'] :
                    date = self.date_string(slot_d['date'].lower() )
                    un_formated_date = event_item.get('date')
                    if self.date_string(un_formated_date) == date :
                        rsp = self.rsp_search_event_time(event_item)
                        found = True
                if slot_d['time']:
                    if  ' '.join(event_item.get('time').split(' ')) == slot_d['time']:
                        rsp = self.rsp_search_event_time(event_item)
                        found = True
        if not found:
            rsp = 'sry ,no relevant events found in database.'
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

# pprint(Navigate(None,None,None).get_city_weather_forecast("Shenzhen"))