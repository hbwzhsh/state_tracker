from dialogue_state.entity_value import EntityValue

class Entity():
    def __init__(self,name,value = None):
        self.type_name = name
        if  value:
            self.entity_value_object = EntityValue(value)
        else:
            self.entity_value_object = EntityValue("")
        self.start = -1
        self.end = -1
        self.index = -1
        self.enable_extend = False
        if self.type_name in self.get_extendble_slot()or self.cut_inform_type()in self.get_extendble_slot():
            self.enable_extend = True

    def allow_extend(self):
        self.enable_extend= True

    def get_extendble_slot(self):
        return ["food2"]

    def copy(self):
        return Entity(self.get_entity_type(),self.get_entity_value())

    def inform_slot(self):
        return Entity('inform_'+self.get_entity_type(),self.get_entity_value())

    def cut_inform_type(self):
        if 'inform_' in self.get_entity_type():
            return self.get_entity_type()[7:]
        else:
            return self.get_entity_type()
    def cut_inform(self):
        if 'inform_' in self.get_entity_type():
            return Entity(self.get_entity_type()[7:], self.get_entity_value())
        return Entity(self.get_entity_type(),self.get_entity_value())

    def set_value(self,value):
        self.entity_value_object.set_value(value)

    def get_entity_type(self):
        return self.type_name

    def get_slot_template(self):
        return '['+ self.type_name +']'

    def get_entity_value(self):
        return self.entity_value_object.entity_value

    def set_start_end(self,start,end):
        self.start = start
        self.end = end

    def is_slot(self,word_left,word_right):
        if word_left >= self.start and word_right <= self.end:
            return True
        return False

    def __str__(self):
        return '{}({})'.format(self.get_entity_value(),self.get_entity_type())

    def print_entity(self):
        print('{}({})'.format(self.get_entity_value(),self.get_entity_type()))

    def get_rasa_word(self):
        if self.get_entity_type() =='O':
            word = self.get_entity_value()
        else:
            word = '[{}]({})'.format(self.get_entity_value(),self.get_entity_type())
        print(word)
        return word

    def get_bert_ner_terms(self):
        word = self.get_entity_value()
        word =  word.strip("\n").split(" ")
        word = [w for w in word if w!=""]
        res = []
        if self.get_entity_type() == 'O':


            for w in word:
                if w =="":continue
                res.append([w,"o"])
        else:
            label = self.get_entity_type()

            for i, w in enumerate(word):

                if i == 0:
                    res.append([w,"B-"+label])
                else:
                    res.append([w, "I-" + label])
        return res