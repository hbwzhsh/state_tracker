from slot_filling_model import SlotModel
import typing
from typing import List, Text, Optional, Dict, Any

#from util_database import sentence_to_template,get_slot_values_dict
import random
import re
class Action():
    def __init__(self,name,d):
        self.action_name = name
        self.utterance = None
        self.utterance_dict = d
        self.template = None
        self.slot_values = None#get_slot_values_dict()
        self.slots = []

    def name(self):
        return self.action_name

    def get_utterance(self):
        print(self.action_name)
        if self.utterance_dict:
            self.utterance = self.utterance_dict[self.action_name]
        if  type(self.utterance) is str:
            return self.utterance
        elif type(self.utterance) is list:
            self.utterance =  self.utterance[random.randint(0,len(self.utterance))]
            self.template = self.utterance
            return self.utterance

    def get_action_temp(self,action):
        action_utterance_dict = {
            'action_ask_area': 'Would you like their location?',
            'action_ask_pricerange': 'Were you looking for an expensive or moderately priced restaurant?',
            'action_ask_food': 'Do you have a cuisine preference?',
            'action_goodbye': 'Goodbye.',
            'action_morehelp': 'Anything else today?',
            'action_inform_address': "their address is [found_address]",
            'action_inform_food': 'They serve [found_food] food .',
            'action_inform_phone': 'Their phone number is [found_phone].',
            'action_inform_area': 'it is located in the [found_area] part of town',
            'action_inform_postcode': 'The postcode is [found_postcode].',
            'action_inform_address_phone':"their address is [found_address] and Their phone number is [found_phone].",
            'action_inform_address_postcode':"their address is [found_address] and Their postcode is [found_postcode].", \
            'action_inform_phone_postcode':"their phone number is [found_phone] and Their postcode is [found_postcode].",
            'action_inform_all': "their phone number is [found_phone] and Their postcode is [found_postcode] and their address is [found_address]."
        }
        return action_utterance_dict.get(action)

    def get_filled_response(self,slot_list):
        # for s in slot_list:
        #     print(s)
        #slot_name = ['area','pricerange','food']
        #slot_name = [slot.get_entity_type() for slot in slot_list]
        #tmp = sentence_to_template(self.template,self.slot_values,slot_name)
        tmp = self.get_action_temp(self.action_name)
        if tmp:
            return self.template_to_utterance(tmp, slot_list)
        return None

    def fill_slot(self,utterance,slot_value):
        # parsing to get slot
        result = self.rasa_nlu_parser.predict(utterance)
        slu_form = self.rasa_nlu_parser.rasa2slu(result)
        # for slot in slu_form:
        #     act = slot['act']
        #     a_slot_name = slot['slots'][0][0]
        #     a_slot_value = slot['slots'][0][1]
        #     if act == 'inform':
        #         q_slot_value

        # check slot set of user query
        # if bot has inform slot ,user has request the same slot ,replace value

        # replace the slot value

        # return sentence
        return ""

    def template_to_utterance(self,template:str,slot_list:list):
        for slot in slot_list:
            if slot.get_slot_template() in template:
                template = template.replace(slot.get_slot_template(),
                                            slot.get_entity_value())
        return template

    def __str__(self) -> Text:
        return "Action('{}')".format(self.name())


