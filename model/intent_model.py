from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter

from collections import Counter
import en_core_web_sm
import time
import json
import datetime
weekdays  =["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
class IntentModel():
    def __init__(self):
        self.model = None
        self.trainer = None
        self.model_path = './slot_filling/'
        self.data_path = './kvret_intent.md'

    def load(self,path = 'projects\\default\\model_20190424-210311'):
        self.model = Interpreter.load(path)
    def load_model_by_name(self,name):
        self.model = Interpreter.load('slot_filling\\default\\{}'.format(name))

    def train(self,training_data):
        self.trainer = Trainer(config.load("nlu_config.yml"))
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

    def predict(self,sentence):
        return self.model.parse(sentence)

    def predict_batch(self,sentences):
        return [self.predict(sent)for sent in sentences]

    def save(self,name = "model"):
        self.trainer.persist(self.model_path,fixed_model_name=name)

    def load_data_and_train(self,name):
        training_data_md = load_data(self.data_path)
        self.train(training_data_md)
        self.save(name)

    def get_intent_vec(self,sentence):
        all_int = ["ask_how_weather", "inform_city", "ask_is_weather", "set_reminder", "ask_reminder", "inform_time",
                   "inform_date", "inform_event", "inform_partner", "ask_place", "ask_navigation", "thank",
                   "ask_address", "yes"]
        if isinstance(sentence,str)==False:
            return [0.0]*len(all_int)
        intent_ranking = self.predict(sentence)["intent_ranking"]
        intent_ranking = {d["name"]:d["confidence"]for d in intent_ranking}
        for name in all_int:
            if name not in intent_ranking.keys():
                intent_ranking[name] = 0.0
        vec  =[intent_ranking[name] for name in all_int]
        return vec
# model = SlotModel()
# # model.load_data_and_train("act_kvret")
# model.load_model_by_name("act_kvret")
# a = model.predict("set a reminder for my dinner")
# print(a)