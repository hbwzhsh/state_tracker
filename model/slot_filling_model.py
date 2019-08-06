from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from dialogue_state.entity import  Entity
from constant import  *
from path_config import *
from loggers import dbot_online_logger
import en_core_web_sm
import time
import json
import datetime
from policy.memory_policy import rasa_md_loader_by_task
weekdays  =["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

class SlotModel():
    def __init__(self,model_path='./slot_filling/', data_path = './data/CamRest.md',model_name = "sf_intent_model"):
        self.model = None
        self.trainer = None
        self.model_path = model_path
        self.data_path = data_path
        self.model_name = model_name

    def load(self,path = 'projects\\default\\model_20190424-210311'):
        self.model = Interpreter.load(path)

    def load_by_model_name(self):
        path = "slot_filling\\default\\{}".format(self.model_name)
        self.model = Interpreter.load(path)

    def train(self,training_data):
        self.trainer = Trainer(config.load(rasa_nlu_config_yml))
        self.trainer.train(training_data)

    def rasa2slu(self,parsing_result):
        entities = parsing_result['entities']
        ds = []
        for e_dict in entities:
            user_act_slot = e_dict['entity']
            slot_value = e_dict['value']
            d = dict()
            d['act'] = user_act_slot.split('_')[0]
            d['slots'] = [[user_act_slot.split('_')[1], slot_value]]
            ds.append(d)
        return ds

    def rasa2slots(self,parsing_result,threshold= 0):
        entities = parsing_result['entities']
        es = []
        for e_dict in entities:
            user_act_slot = e_dict['entity']
            slot_value = e_dict['value']
            e = Entity(user_act_slot, slot_value)
            confidence = e_dict['confidence']
            if confidence <threshold:
                continue
            else:
                dbot_online_logger.debug("from rasa2slots:added the high confidence slot {},{},{}".format(user_act_slot,slot_value,confidence))
                es.append(e)
        return es

    def is_have_uncertain_slot(self,parsing_result):
        entities = parsing_result['entities']
        need_2_confirm = [e_dict for e_dict in entities if e_dict["confidence"] < slot_threshold]
        if need_2_confirm==[]:
            return False
        return True




    def predict(self,sentence):
        return self.model.parse(sentence)

    def get_valid_intent_from_domain(self,task,sentence):
        story_ds,intent_list,action_list = rasa_md_loader_by_task(task)
        intent_ranking = self.predict(sentence).get("intent_ranking")
        if intent_ranking:
            valid_intent = [d for d in intent_ranking if d["name"] in intent_list]
            return valid_intent
        return None

    def get_intent_under_domain(self,task):
        story_ds, intent_list, action_list = rasa_md_loader_by_task(task)
        return intent_list

    def get_domain_intent_from_q(self,q,task):
        intents = self.get_intent_under_domain(task)
        for intent in intents:
            if intent in q:
                return intent
        return None


    def is_have_uncertain_intent(self,task,sentence):
        valid_intents = self.get_valid_intent_from_domain(task,sentence)
        if not valid_intents:
            return True
        intent,max_confidence = self.get_max_intent(valid_intents)
        if max_confidence <intent_threshold:
            return  True
        return False

    def intent_affirm(self,task,sentence):
        """
        return the intent to confirm


        :param task:
        :param sentence:
        :return:
        """
        res = "sorry,we cannot understand your intent,\n " \
              "the possible intent will shown below,please input the right intent to confirm:\n"
        valid_intents = self.get_valid_intent_from_domain(task,sentence)
        if not valid_intents:
            possible = self.get_intent_under_domain(task)
            res+=",".join(possible)
            return res,possible
        intent,confidence = self.get_max_intent(valid_intents)
        if confidence <intent_threshold:
            possible = [d["name"] for d in valid_intents if d["confidence"]<intent_threshold]
            res+=",".join(possible)
        else:
            return None,None
        return res,possible

    def get_max_intent(self,intents):
        confidence_max = 0
        intent = None
        if not intents:
            return None,0
        for d in intents:
            conf = d["confidence"]
            intent_name = d["name"]
            if conf > confidence_max:
                confidence_max = conf
                intent = intent_name
        return intent,confidence_max

    def rasa_intent_with_domain(self,sentence,task):
        if not task:
            return "undefined"
        valid_intent = self.get_valid_intent_from_domain(task,sentence)
        if not valid_intent:
            return "undefined"
        print(valid_intent)
        intent, confidence_max = self.get_max_intent(valid_intent)
        return intent

    def predict_batch(self,sentences):
        return [self.predict(sent)for sent in sentences]

    def save(self,name = "model"):
        self.trainer.persist(self.model_path,fixed_model_name=name)

    def load_data_and_train(self):
        training_data_md = load_data(self.data_path)
        self.train(training_data_md)
        self.save(self.model_name)

def default_train_rasa_nlu():
    task_name = "all"
    #new_model = SlotModel(data_path="C://demo4rasa//data//nlu_{}.md".format(task_name),model_name=task_name)
    new_model = SlotModel(model_path=rasa_nlu_model_path_save,data_path = rasa_nlu_data_path,model_name=task_name)
    new_model.load_data_and_train()

#running rasa nlu training for the example domains
#default_train_rasa_nlu()




# new_model.load_by_model_name()
# us = ["cheeseburger please","I want hot tea","I want ice tea","Hot Chocolate","i want fries"]
# for u in us:
#     r = new_model.predict(u)#["intent"]
#     r = new_model.rasa_intent_with_domain(u,"order_food")#["intent"]
#     print(r)
class kvret_sf_intent(SlotModel):
    #fixed_model_name
    def __init__(self):
        self.model = None
        self.trainer = None
        self.model_path = './kvret_sf/'
        self.data_path = './kvret_sf/intent_kvret_train.md'
        self.model_name = "kvret"

    def load_data_and_train(self):
        training_data_md = load_data(self.data_path)
        self.train(training_data_md)
        self.save(self.model_name)

useful_slots = ['time', 'distance', 'event', 'party', 'location', 'date', 'agenda', 'poi_type', 'address', 'room',
                'poi', 'weather_attribute']

class GeneralNer():
    def __init__(self):
        pass

    @staticmethod
    def get_week_date(sentence):
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for w in weekdays:
            if w in sentence.lower():
                return w
        return None

    @staticmethod
    def get_month(sentence):
        months = ["January", "February", "March", "April", "May",
                  "June", "July", "August", "September", "October", "Novemeber", "December"]
        for w in months:
            if w in sentence.lower():
                return w
        return None

    @staticmethod
    def get_phone_number(text):
        import re
        numbers = re.findall(r'\d+', text, re.MULTILINE)
        if len(numbers) == 0:
            return None
        else:
            max_number = 0
            for n in numbers:
                n = int(n)
                if n > max_number:
                    max_number = n
            return max_number

    @staticmethod
    def get_phone_number_entity(text):
        number = GeneralNer.get_phone_number(text)
        if number and len(str(number))>3:
            return Entity("number",number)
        return None

    @staticmethod
    def get_basic_entity(sentence):
        print("===running basic ner model===")
        es = []
        nlp = en_core_web_sm.load()
        doc = nlp(sentence)
        for  X in doc.ents:
            print(X)
            if X.label_ == "DATE":
                if (":" in X.text) or ("am" in X.text) or ("pm" in X.text):
                    e = Entity("time", X.text)
                else:
                    e = Entity("date", X.text)
                es.append(e)
            if X.label_ == "GPE":
                e = Entity("location", X.text)
                es.append(e)
            if X.label_ == "CARDINAL":
                e = Entity("time", X.text)
                es.append(e)
            if X.label_ == "TIME":
                e = Entity("time", X.text)
                es.append(e)
        phone_number = GeneralNer.get_phone_number_entity(text=sentence)
        if phone_number:
            es.append(phone_number)
        for e in es:
            print(e)
        return es

class CamRestSlotFilling():
    def __init__(self):
        self.informable = {
        "area" : ["centre","north","west","south","east"],
        "food" : ["afghan","african","afternoon tea","asian oriental","australasian","australian","austrian","barbeque","basque","belgian","bistro","brazilian","british","canapes","cantonese","caribbean","catalan","chinese","christmas","corsica","creative","crossover","cuban","danish","eastern european","english","eritrean","european","french","fusion","gastropub","german","greek","halal","hungarian","indian","indonesian","international","irish","italian","jamaican","japanese","korean","kosher","latin american","lebanese","light bites","malaysian","mediterranean","mexican","middle eastern","modern american","modern eclectic","modern european","modern global","molecular gastronomy","moroccan","new zealand","north african","north american","north indian","northern european","panasian","persian","polish","polynesian","portuguese","romanian","russian","scandinavian","scottish","seafood","singaporean","south african","south indian","spanish","sri lankan","steakhouse","swedish","swiss","thai","the americas","traditional","turkish","tuscan","unusual","vegetarian","venetian","vietnamese","welsh","world"],
        "pricerange" : ["cheap","moderate","expensive"]
    }
    def get_all_value_by_slot(self,slotname):
        res =  self.informable.get(slotname)
        if not res:
            res = []
        return res


    def find_slot_by_type(self,slot_type,user_q):
        for value in self.get_all_value_by_slot(slot_type):
            if value.lower() in user_q.lower():
                return Entity(slot_type,value)
        return None

    def find_all_types_slot(self,user_q):
        slots = []
        for slot_type in ["area","food","pricerange"]:
            slot = self.find_slot_by_type(slot_type,user_q)
            if slot:
                slots.append(slot)
        return slots




class HelperNER():

    def __init__(self):
        file = open('C://kvret_entities.json','rb')
        self.ners_dict = json.loads(file.read())
        file.close()


    def get_poi_data(self):
        return self.ners_dict.get('poi')

    def get_all_poi_name(self):
        return [u.get('poi') for u in self.get_poi_data()]



    def get_entity(self,sentence):
        weather_attribute =  ["lowest temperature", "highest temperature", "overcast",
                              "snow", "stormy", "hail",
                              "hot", "rain", "cold",
                              "clear skies", "cloudy", "warm", "windy",
                              "foggy", "humid", "frost", "blizzard", "drizzle", "dry", "dew", "misty"
                              ]
        es = []
        #decide whether clearify or check
        if "what" in sentence and "weather" in sentence:
            e = Entity("question_word","what weather")
            es.append(e)
        else:
            for weather in weather_attribute:
                if weather in sentence.lower():
                    if ("if" in sentence) or ("whether" in sentence) or ("will" in sentence):
                        e = Entity("question_word", "if weather")
                        es.append(e)
        #
        nlp = en_core_web_sm.load()
        doc = nlp(sentence)
        print([(X.text, X.label_) for X in doc.ents])

        for  X in doc.ents:
            if X.label_ == "DATE":
                if (":" in X.text) or ("am" in X.text) or ("pm" in X.text):
                    e = Entity("time", X.text)
                else:
                    e = Entity("date", X.text)
                es.append(e)
            if X.label_ == "GPE":
                e = Entity("location", X.text)
                es.append(e)
            if X.label_ == "CARDINAL":
                e = Entity("time", X.text)
                es.append(e)
            if X.label_ == "TIME":
                e = Entity("time", X.text)
                es.append(e)
        for entity_type,values in self.ners_dict.items():
            for v in values:
                if str(v).lower() in sentence.lower():
                    e = Entity(entity_type,v)
                    es.append(e)
        for poi_name in self.get_all_poi_name():
            if poi_name:
                if str(poi_name).lower() in sentence.lower():
                    e = Entity('poi_name', v)
                    es.append(e)
        for e in es:
            print(e)
        return es

    @staticmethod
    def entity2vector(entitys):
        vec = [0]*len(useful_slots)
        for e in entitys:
            for s in useful_slots:
                if e.get_entity_type() == s:
                    vec[useful_slots.index(s)] =1
        return vec

    @staticmethod
    def get_current_time():
        today_date = time.strftime("%d", time.localtime())
        hour = time.strftime("%h", time.localtime())
        month = time.strftime("%m", time.localtime())
        today_date = int(today_date)
        hour = int(hour)
        month = int(month)
        return today_date,hour,month

    @staticmethod
    def get_current_date(get_str = False):
        if get_str:
            return str(datetime.date.today())
        return datetime.date.today()



    @staticmethod
    def next_date(n):
        return str(HelperNER.get_current_date() +datetime.timedelta(days=n))

    @staticmethod
    def tomorrow():
        return str(HelperNER.get_current_date() +datetime.timedelta(days=1))

    @staticmethod
    def day_after_tomorrow():
        return str(HelperNER.get_current_date() +datetime.timedelta(days=2))

    @staticmethod
    def date2day(date):
        return int(str(date).split("-")[-1])

    @staticmethod
    def date2month(date):
        return int(str(date).split("-")[-2])

    @staticmethod
    def weekday2date(weekday):
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        if weekday not in weekdays:
            print("weekday{}  should be inside the weekdays {}".format(weekday,weekdays))
            return

        today_wd = datetime.datetime.today().weekday()
        weekday_index = weekdays.index(weekday)
        if weekday_index <today_wd:
            d = weekday_index + 7 - today_wd
            day = HelperNER.date2day(HelperNER.next_date(d))
            month = HelperNER.date2month(HelperNER.next_date(d))
        else:
            d = weekday_index - today_wd
            day = HelperNER.date2day(HelperNER.next_date(d))
            month = HelperNER.date2month(HelperNER.next_date(d))
        return month,day

    @staticmethod
    def get_next_n_days(n):
        ds =[]
        for i in range(n):
            d = HelperNER.date2day(HelperNER.next_date(i))
            ds.append(d)
        return ds
        # sentence = 'I want to find a restaurant that serves chinese food and located at south of the town'
# slot = SlotModel()
# #model.load_data_and_train()
# slot.load()
# slot.predict(sentence)

# - Where is the nearby (hospital)[poi_type] ?
# - Give me the address to (Stanford Childrens Health)[poi] .
# - Okay, thanks
# - show me the (closest)[distance] location where i can get (chinese food)[poi_type]


# print(a,b)
# a = kvret_sf_intent()
# a.load(a.model_path+"\\default\\"+a.model_name)
#
#

#
# result = a.predict("what is the weather like tomorrow and Monday,friday in Shenzhen")
# s = "show me the closest location where i can get chinese food"
# s = "goodbye"
# result = a.predict(s)
# print(result)
class NER():
    @staticmethod
    def find_slot_from_slotlist(target,slotlist):
        for slot in slotlist:
            if slot.get_entity_type() == target.get_entity_type() and slot.get_entity_value()==target.get_entity_value():
                return slot
        return None

    @staticmethod
    def merge(slots1,slots2):
        new_slots = []
        for s in slots1+slots2:
            if not NER.find_slot_from_slotlist(s,new_slots):
                new_slots.append(s)
        return new_slots

    @staticmethod
    def slot_collect(slotlist_list):
        last = []
        for slotlist in slotlist_list:
            last = NER.merge(last,slotlist)
        return last

    @staticmethod
    def multi_ner_tool(tools,sentence):
        results = [t.parse(sentence) for t in tools]
        return NER.slot_collect(results)
print(HelperNER.next_date(1))