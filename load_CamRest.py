import json
informable = ['area','food','pricerange']
requestable = informable + ['phone', 'postcode', 'address']
prefix_info = ['inform_' + u for u in informable]
prefix_request = ['request_' + u for u in requestable]
## 'action_inform_address_phone','action_inform_address_postcode'
## 'action_inform_phone_postcode'
# if 'action_inform_address' in action_from_sentence and 'action_inform_phone'\
#         in action_from_sentence and 'action_inform_postcode' in action_from_sentence:
#     action_from_sentence.remove('action_inform_address')
#     action_from_sentence.remove('action_inform_phone')
#     action_from_sentence.remove('action_inform_postcode')
#     action_from_sentence.append('action_inform_all')
# if 'action_inform_address' in action_from_sentence and 'action_inform_phone' in action_from_sentence:
#     action_from_sentence.remove('action_inform_address')
#     action_from_sentence.remove('action_inform_phone')
#     action_from_sentence.append('action_inform_address_phone')
#
# if 'action_inform_address' in action_from_sentence and 'action_inform_postcode' in action_from_sentence:
#     action_from_sentence.remove('action_inform_address')
#     action_from_sentence.remove('action_inform_postcode')
#     action_from_sentence.append('action_inform_address_postcode')
#
# if 'action_inform_phone' in action_from_sentence and 'action_inform_postcode' in action_from_sentence:
#     action_from_sentence.remove('action_inform_phone')
#     action_from_sentence.remove('action_inform_postcode')
#     action_from_sentence.append('action_inform_phone_postcode')
informable = ['area','food','pricerange']

more_actions = ['action_goodbye','action_morehelp','action_inform_address',\
                'action_inform_food','action_inform_phone',\
                'action_inform_area','action_inform_postcode'
                ]
act_response =['action_inform_phone', 'action_inform_area', 'action_inform_pricerange',\
               'action_inform_food', 'action_inform_postcode', 'action_inform_address','action_goodbye','action_morehelp']


def get_dialogue_content():
    json_file = open('C:\CamRest676.json',mode='r',encoding='utf-8')
    dialogue_content =json.load(json_file)
    return dialogue_content
dialogue_content =   get_dialogue_content()


def get_dialogue(id):
    return dialogue_content[id]['dial']

def get_slu(id,turn):
    if len(get_dialogue(id)) <= turn:return None
    return get_dialogue(id)[turn]['usr']['slu']

def get_transcript(id,turn):
    if len(get_dialogue(id)) <= turn: return None
    return get_dialogue(id)[turn]['usr']['transcript']

def get_request(id):
    return dialogue_content[id,]['goal']['request-slots']

def get_bot_ask_label(id,turn):
    if len(get_dialogue(id)) <= turn: return None
    return get_dialogue(id)[turn]['sys']['DA']

def slot_state_dict(informable,request):
    d = dict()
    for slot in request:
        d['request_'+slot] = 0
    for slot in informable:
        d['inform_'+slot] = 0
    return d

def dict_to_feature(d,slot_names):
    return [d[slot] for slot in slot_names]

def get_feature_from_nlu(slu):
    states = slot_state_dict(informable, informable + ['phone', 'postcode', 'address'])
    if slu != None:
        for user_act in slu:
            if user_act['act'] == 'request':
                slot_name = user_act['slots'][0][1]
                if 'request_' + slot_name in states.keys():
                    states['request_' + slot_name] = 1
            if user_act['act'] == 'inform':
                slot_name = user_act['slots'][0][0]
                # if slot_name not in informable+['phone','postcode','address']:
                #     print(slot_name)
                if 'inform_' + slot_name in states.keys():
                    states['inform_' + slot_name] = 1
    state_feature = dict_to_feature(states,prefix_request + prefix_info)
    return state_feature

def get_label_from_sentence(sent):
    sent = sent.lower()
    labels = []
    if 'welcome' in sent or 'goodbye' in sent or ' nice day' in sent:
        labels.append('action_goodbye')
    if 'anything else' in sent:
        labels.append('action_morehelp')

    # if 'there address is' in sent or 'their address is' in sent:
    #     labels.append('action_inform_address')
    #
    # if 'they serve' in sent:
    #     labels.append('action_inform_food')
    #
    # if 'phone number is ' in sent:
    #     labels.append('action_inform_phone')
    #
    # if ' located at ' in sent or 'located in' in sent:
    #     labels.append('action_inform_area')
    #
    # if 'postcode is' in sent:
    #     labels.append('action_inform_postcode')
    return labels

def is_a_search(bot):
    bot = bot.lower()
    kw = ['several restaurant','there are ','meet your ','no restaurant','no record','matching','your request','meeting your criteria']
    for w in kw:
        if w in bot:
            return True

def get_X_Y_from_raw_text():
    X = []
    data = []
    Y = []
    state_his_s =[]
    text_historys = []
    for dial_id in range(len(dialogue_content)):
        conversation = dialogue_content[dial_id]
        state_his = []
        text_history = []
        for turn in range(len(get_dialogue(dial_id))):
            user_query = get_dialogue(dial_id)[turn]['usr']['transcript'].lower()
            bot_reply = get_dialogue(dial_id)[turn]['sys']['sent'].lower()
            slu = get_slu(dial_id, turn)
            request_slots = []
            states = slot_state_dict(informable, informable + ['phone', 'postcode', 'address'])
            if slu != None:
                constraint = []
                for user_act in slu:
                    if user_act['act'] == 'request':
                        slot_name = user_act['slots'][0][1]
                        if 'request_' + slot_name in states.keys() and slot_name.lower() in user_query.lower():
                            states['request_' + slot_name] = 1
                            request_slots.append(slot_name)
                    if user_act['act'] == 'inform':
                        slot_name = user_act['slots'][0][0]
                        slot_value = user_act['slots'][0][1]
                        constraint.append(slot_value)
                        # if slot_name not in informable+['phone','postcode','address']:
                        #     print(slot_name)
                        if 'inform_' + slot_name in states.keys():
                            states['inform_' + slot_name] = 1
            state_feature = dict_to_feature(states,prefix_request + prefix_info)
            labels = []
            X.append(state_feature)
            state_his.append(state_feature[:])
            state_his_s.append(state_his[:])
            text_history.append(user_query[:])
            text_history.append(bot_reply[:])
            text_historys.append(text_history[:])
            y = [0, 0, 0]
            y_more_action = [0] * len(more_actions)
            bot_actions = get_dialogue(dial_id)[turn]['sys']['DA']
            for action in bot_actions:
                if action in ['area', 'pricerange', 'food']:
                    y[['area', 'pricerange', 'food'].index(action)] = 1

            action_from_sentence = get_label_from_sentence(bot_reply)
            inform_action = ['action_inform_' + slot for slot in request_slots]



            for action in action_from_sentence:
                if action in more_actions:
                    labels.append(action)
                    y_more_action[more_actions.index(action)] = 1
            y.extend(y_more_action)
            labels.extend(inform_action)


            is_search = False
            for slot in constraint:
                if slot.lower() in bot_reply.lower():
                    is_search = True
                    break
            if is_search or is_a_search(bot_reply):
                y.append(1)
                labels.append('action_search_rest')
            else:
                y.append(0)
            Y.append(y)
            data.append([state_his,labels])
            #print(labels,user_query,bot_reply)
    return X,Y,data,state_his_s,[d[1] for d in data],text_historys

def get_action_template(dialogue_content):
    X = []
    Y = []
    y_action_labels = []
    utterances = []
    for dial_id in range(len(dialogue_content)):
        conversation = dialogue_content[dial_id]
        for turn in range(len(get_dialogue(dial_id))):
            slu = get_slu(dial_id, turn)
            states = slot_state_dict(informable, informable + ['phone', 'postcode', 'address'])
            if slu != None:

                for user_act in slu:
                    if user_act['act'] == 'request':
                        slot_name = user_act['slots'][0][1]
                        if 'request_' + slot_name in states.keys():
                            states['request_' + slot_name] = 1
                    if user_act['act'] == 'inform':
                        slot_name = user_act['slots'][0][0]
                        # if slot_name not in informable+['phone','postcode','address']:
                        #     print(slot_name)
                        if 'inform_' + slot_name in states.keys():
                            states['inform_' + slot_name] = 1

            state_feature = dict_to_feature(states,prefix_request + prefix_info)
            X.append(state_feature)
            y = [0, 0, 0]
            y_more_action = [0] * len(more_actions)
            y_action_label = []
            bot_ask_action = get_dialogue(dial_id)[turn]['sys']['DA']
            for action in bot_ask_action:
                if action in ['area', 'pricerange', 'food']:
                    y[['area', 'pricerange', 'food'].index(action)] = 1
                    y_action_label.append('action_ask_'+action)
            action_from_sentence = get_label_from_sentence(get_dialogue(dial_id)[turn]['sys']['sent'])
            for action in action_from_sentence:
                if action in more_actions:
                    y_more_action[more_actions.index(action)] = 1
                    y_action_label.append(action)
            utterance = get_dialogue(dial_id)[turn]['sys']['sent']
            utterances.append(utterance)
            y.extend(y_more_action)
            Y.append(y)
            y_action_labels.append(y_action_label)
    return utterances,y_action_labels
#get_X_Y_from_raw_text()


# more_actions = ['action_goodbye', 'action_morehelp', 'action_inform_address', \
#                 'action_inform_food', 'action_inform_phone', \
#                 'action_inform_area', 'action_inform_postcode']
all_action = ['action_ask_area', 'action_ask_pricerange', 'action_ask_food'] + more_actions
#utterances,y_action_labels = get_action_template(dialogue_content)
def build_action_utterance(utterances,y_action_labels):
    action_temp_dict = dict()
    for action in all_action:
        action_temp_dict[action] = []
    for utterance,y_action_label in zip(utterances,y_action_labels):
        if len(y_action_label) <= 2 :
            for label in y_action_label:
                action_temp_dict[label].append(utterance)
            #action_temp_dict[y_action_label[0]].append(utterance)
    return action_temp_dict

i = 3
i = 4



#
get_X_Y_from_raw_text()
# import pickle
# file = open('camrest_action_learning.pk','wb')
# pickle.dump([X,Y,data],file)
# file.close()