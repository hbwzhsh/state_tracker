from rasa_nlu.training_data import load_data
from rasa_nlu.model import Trainer
from rasa_nlu import config
from rasa_nlu.model import Interpreter
from entity import  Entity
import json
class SlotModel():
    def __init__(self):
        self.model = None
        self.trainer = None
        self.model_path = './slot_filling/'
        self.data_path = './data/CamRest.md'

    def load(self,path = 'projects\\default\\model_20190424-210311'):
        self.model = Interpreter.load(path)

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

    def save(self):
        self.trainer.persist(self.model_path)

    def load_data_and_train(self):
        training_data_md = load_data(self.data_path)
        self.train(training_data_md)
        self.save()

class HelperNER():
    def __init__(self):
        file = open('D://conv_data//kvret_dataset_public//kvret_entities.json','rb')
        self.ners_dict = json.loads(file.read())
        file.close()

    def get_entity(self,sentence):
        es = []
        for entity_type,values in self.ners_dict.items():
            for v in values:
                if str(v).lower() in sentence.lower():
                    e = Entity(entity_type)
                    e.set_value(v)
                    es.append(e)
        return es





        # sentence = 'I want to find a restaurant that serves chinese food and located at south of the town'
# slot = SlotModel()
# #model.load_data_and_train()
# slot.load()
# slot.predict(sentence)