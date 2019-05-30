class EntityValue():
    def __init__(self, name):
        self.entity_value = name
        self.entity = None

    def set_value(self, value):
        self.entity_value =value


    def get_entity_type(self):
        return self.entity.get_type()

    def set_entity_type(self,type):
        self.entity.set_type(type)

    def get_entity_value(self):
        return self.string_value

