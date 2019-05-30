from entity import Entity
class SlotTool():
    def __init__(self):
        self.sentence = None
        self.words = None
        self.slots = None
        self.delimiter = ' '


    def words(self):
        return self.sentence.split()

    def find_slot_by_type(self,slot_type):
        for slot in self.slots:
            if slot.get_entity_type() == slot_type:
                return slot
        return None

    def get_slot_value_pair(self):
        pass

    def set_words(self,words):
        self.words = words

    def set_slots(self,slots):
        self.slots = slots

    def set_words_by_sentence(self):
        self.words = self.sentence.split(self.delimiter)

    def build_slot_list_from_words_slots(self,words,slot_label):
        slot_list = []
        for word, label in zip(words, slot_label):
            slot = Entity(label)
            slot.set_value(word)
            slot_list.append(slot)
        return  slot_list

    def camrest_to_slots(self,slu):
        slot_list = []
        for slu_event in slu:
            if slu_event['act'] == 'inform':
                slot_name = slu_event['slots'][0][0]
                slot_value = slu_event['slots'][0][1]
                slot = Entity(slot_name)
                slot.set_value(slot_value)
                slot_list.append(slot)
        self.slots = slot_list
        return  slot_list

    def rasa_to_pair(self,sentence,parsing_result):
        entities = parsing_result['entities']
        ds = []
        slots = []
        for e_dict in entities:
            slot_type = e_dict['entity']
            slot_value = e_dict['value']
            start = e_dict['start']
            end = e_dict['end']
            slot = Entity('O')
            slot.set_value(slot_value)
            slot.type_name = slot_type
            slot.set_start_end(start,end)
            slots.append(slot)
        word_start,word_end = 0,0
        labels = []
        for i,char in enumerate(sentence + ' '):
            word_end = i
            if char == ' ':
                word_is_slot = False
                for slot in slots:
                    if slot.is_slot(word_start,word_end):
                        word_is_slot = True
                        slot.set_value(sentence[word_start:word_end])
                        labels.append(slot)
                if word_is_slot == False:
                    unlabeled = Entity('O')
                    unlabeled.set_value(sentence[word_start:word_end])
                    labels.append(unlabeled)
                word_start = i+1
        # for s in labels:
        #     s.print_entity()
        # print(' '.join([s.get_rasa_word() for s in labels]))
        return labels

    @staticmethod
    def slotset2dict(slotset):
        d = dict()
        for slot in slotset:
            d[slot.get_entity_type()] = slot.get_entity_value()
        return d

    @staticmethod
    def dict2slotset(slot_dict):
        slotset = []
        for k,v in slot_dict.items():
            slot = Entity(k,v)
            slotset.append(slot)
        return slotset

    def __str__(self):
        if not self.slots:
            return "no slots"
        else:
            r = ''
            for slot in self.slots:
                r += " {}({})\n ".format(slot.get_entity_value(),slot.get_entity_type())
            return r

#print(SlotSentence.dict2slotset({'food':'chinese'}))