from model.action_model import Glove
from data_converter.babi import read_Babi
import numpy as np
from keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from data_converter.babi import get_Babi_label
class Babi(Glove):
    def get_tb(self):
        FILE_DIR = "c://tb"
        tb = TensorBoard(log_dir=FILE_DIR,  # 日志文件保存位置
                         histogram_freq=1,  # 按照何等频率（每多少个epoch计算一次）来计算直方图，0为不计算
                         batch_size=32,  # 用多大量的数据计算直方图
                         write_graph=True,  # 是否在tensorboard中可视化计算图
                         write_grads=False,  # 是否在tensorboard中可视化梯度直方图
                         write_images=False,  # 是否在tensorboard中以图像形式可视化模型权重
                         update_freq='batch')  # 更新频率
        return tb

    def get_babi_actions(self):
        labels= get_Babi_label()
        return labels

    def get_next_api_call(self,replys,from_i):
        for i,r in enumerate(replys):
            if i < from_i:
                continue
            if "api_call" in r:
                words = r.strip("\n").split(" ")
                _,cuisine,location,price = words
                return cuisine,location,price
        return "*###*@","*###*@","*###*@"

    def vector_or(self,v1,v2):
        res = []
        for l,r in zip(v1,v2):
            if l==1 or r==1:
                res.append(1)
            else:
                res.append(0)
        return res

    def prepare_babi(self):
        labels, all_row = read_Babi()
        many_dialogues = []
        utters_for_dialogues = []
        old_number = 0
        dialogue = []
        utters_for_dialogue = []
        for row in all_row:
            number, history, user_q, sentence, label = row
            number = int(row[0])
            #print(".number..")
            if number>old_number:
                dialogue.append(row)
                utters_for_dialogue.append(user_q)
                utters_for_dialogue.append(sentence)
                old_number = number
            else:
                old_number = number
                many_dialogues.append(dialogue[:])
                #print("len of utter for dialogue ",len(utters_for_dialogue))
                utters_for_dialogues.append(utters_for_dialogue[:])
                utters_for_dialogue = []
                dialogue = []
        historys = []
        intent_vecs = []
        label_vecs = []
        last_action_vecs = []
        utterRNN_vecs = []
        his_data = []

        for dia in utters_for_dialogues:
            turn_vector = []
            for i,utter in enumerate(dia):
                seqs = self.utter_vec(utter)
                turn_vector += seqs
            #print("turn vec length should be ",self.MAX_SENTENCE_LENGTH*len(dia),"but it is ",len(turn_vector))
            for i in range(len(dia)//2):
                end = i*2+1
                if self.turn_number > end:
                    padding_turns = self.turn_number - end
                    vec = padding_turns*self.MAX_SENTENCE_LENGTH*[0] + turn_vector[:self.MAX_SENTENCE_LENGTH * end]
                else:
                    vec = turn_vector[self.MAX_SENTENCE_LENGTH * (end-self.turn_number):self.MAX_SENTENCE_LENGTH * end]
                #print(len(vec))
                his_data.append(vec)
        for dialogue in many_dialogues:
            #print("length of dialogue = ",len(dialogue))
            replys = [turn[-2]for turn in dialogue]
            last_filling_state = [0,0,0]
            for i,turn in enumerate(dialogue):

                last_action = ""
                if i != 0:
                    last_action = dialogue[i-1][-1]
                cuisine,location,price = self.get_next_api_call(replys,i)


                new = turn+[last_action]
                number,history,user_q,sentence,label,last_action = new
                filling_state = {"cuisine":False,"location":False,"price":False}

                if cuisine.lower() in user_q.lower():
                    filling_state["cuisine"] = True
                if location.lower() in user_q.lower():
                    filling_state["location"] = True
                if price.lower() in user_q.lower():
                    filling_state["price"] = True

                #print(user_q,filling_state)
                label_vec = self.one_hot(label,labels)
                last_action_vec = self.one_hot(last_action,labels)
                current_filling_state = self.dict2vector(filling_state,["cuisine","location","price"])
                combine = self.vector_or(current_filling_state,last_filling_state)
                tmp = last_action_vec.tolist() +(current_filling_state + combine)
                tmp = np.array(tmp)
                last_action_vec = tmp
                last_filling_state = combine
                label_vecs.append(label_vec)
                last_action_vecs.append(last_action_vec)
                for j, h in enumerate(history):
                    if not isinstance(h, str):
                        history[j] = ""
                historys.append(history)
        last_action_vecs = np.array(last_action_vecs)
        label_vecs = np.array(label_vecs)
        array4utt_his = np.array(his_data)
        #print(array4utt_his.shape)
        #array4utt_his = self.get_utterance(historys)
        return array4utt_his,last_action_vecs,label_vecs

    def dict2vector(self,d,keys):
        r = []
        for k in keys:
            filling = d[k]
            if filling:
                r.append(1)
            else:
                r.append(0)
        return r

    def map_feature_length(self):
        map_dict = {"intent": len(self.get_babi_actions())+6}
        return map_dict

    def load_data_and_train(self):
        self.model_path = "C://model/pretrain_babi_lastact.h5"
        self.all_action = self.get_babi_actions()
        array4utt_his,last_action_vecs,label_vecs = self.prepare_babi()
        self.build_binary_model(False, add_last_action=False,features=["intent"])
        y = np.array(label_vecs).reshape((-1, len(self.all_action)))
        Xtrain, Xtest, ytrain, ytest, intent_train, intent_test = train_test_split(array4utt_his, y, last_action_vecs,
                                                                                   test_size=0.1,
                                                                                   random_state=42)
        self.model.fit([Xtrain, intent_train], ytrain, batch_size=self.batch_size, verbose=2,
                    epochs=30,
                    validation_data=([Xtest, intent_test], ytest), callbacks=[self.get_tb()])

        self.save(self.model_path)
        self.evaluate([Xtest, intent_test], ytest)

    def load_model_and_evaluate(self):
        self.model_path = "C://model/pretrain_babi_lastact.h5"
        self.all_action = self.get_babi_actions()
        array4utt_his,last_action_vecs,label_vecs = self.prepare_babi()
        y = np.array(label_vecs).reshape((-1, len(self.get_babi_actions())))
        Xtrain, Xtest, ytrain, ytest, intent_train, intent_test = train_test_split(array4utt_his, y, last_action_vecs,
                                                                                   test_size=0.1,
                                                                                   random_state=42)
        self.load_model(self.model_path)
        self.evaluate([Xtest, intent_test], ytest)

#Babi().prepare_babi()
model = Babi()
#model.prepare_babi()
model.load_data_and_train()
#model.load_model_and_evaluate()