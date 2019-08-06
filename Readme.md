## 依赖安装
安装rasa nlu

pip install rasa_nlu[tensorflow]

安装c++ build tool

https://visualstudio.microsoft.com/zh-hans/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16
 
安装spacy

pip install -U spacy

python -m spacy download en_core_web_sm

安装mongo数据库lib

pip install pymongo

## 文件组织结构
```
C:.
│  constant.py  常数
│  loggers.py   debug用的logger
│  nlu_config.yml rasa nlu配置文件
│  path_config.py  数据及模板路径定义
│  Readme.md
│
│
├─action_api    skill相关的api和类
│  │  sample_data.py  demo样例数据
│  │  schdule.py      日程类，用于日程任务  
│  │  __init__.py
│  │
│  └─__pycache__
│         
│
├─data_converter
│  │  Babi.py          babi数据导入相关代码
│  │  __init__.py
│  │
│  └─__pycache__
│         
│
├─data_loader         action分类器相关代码和训练数据
│   
│
├─dialogue_state      管理对话中状态 涉及词槽，事件，数据库，Action类等
│     action.py         Action基类，用于输出bot回复
│     Datebase.py       操作mongodb
│     domain.py         nlu模块和
│     entity.py         slot类
│     entity_value.py   slot值
│     event.py          事件，用来标准化user和bot的消息及nlu信息存储
│     reader.py         配置文件读取
│     slot_tool.py      静态slot相关处理函数
│     state_tracker.py  用户追踪器，用来存放user和bot产生的event和当前slot      
│     task_action.py     Action的子类，有一些特定skill的实现
│     TrackerManager.py  管理所有tracker
│     __init__.py
│   

│
├─kvret_sf
│  │  intent_kvret_train.md
│  │  intent_kvret_train.txt
│  │
│  └─default
│
├─model
│  │  action_model.py      action分类器
│  │  Babi_model.py        action分类器
│  │  intent_model.py
│  │  slot_filling_model.py  rasa nlu模块
│  │  __init__.py
│  │
│  ├─slot_filling
│
├─ner_util                 规则型ner提取
│     expression_test.py
│     spacy_test.py
│     __init__.py
│   
│
├─Policy                     策略，决定bot回复的action/utter标签
│  │  Agent.py               基类
│  │  FormPolicy.py          输入slot名列表，依次引导用户填充slot
│  │  MemoryAgent.py         根据预定义任务型流程  决定回复标签
│  │  MemoryPolicy.py
│  │  __init__.py
│  │
│  │
│  ├─logs
│  │      dbot_offline.log
│  │      dbot_online.log
│  │
│  ├─slot_filling
│  
│
├─server
│      CamRest_sever.py      http服务
│      main.py
│      __init__.py
│
├─slot_filling
│
├─template                  模板和配置
│      action_tmp.txt        utter模板定义文件
│      stories_book_course.md    任务流程定义文件
│      stories_buy_tv.md
│      stories_find_apartment.md
│      stories_order_food.md
│      stories_reminder.md
│      stories_weather.md
│      task_book_course.json    任务所需槽位定义文件
│      task_buy_tv.json
│      task_config.json         当前任务定义
│      task_find_apartment.json
│      task_order_food.json
│      task_reminder.json t
│      新建文本文档.txt
│
├─trainingdata              
       nlu_all.md               rasa nlu 训练数据
```

## 训练nlu model

default train rasa nlu 训练nlu模型并保存

保存完后才能使用MemoryAgent
```
#running rasa nlu training for the example domains
default_train_rasa_nlu()
```

## 测试说明
在template/task_config中输入task名 (buy_tv,book_course,order_food,reminder,weather,find_apartment)

在constant.py中设置slot 和intent的最低置信度

训练nlu模型

运行Policy下的memoryAgent

当slot或intent的置信度低时，bot做澄清操作，user需要输入slot和intent 的名称来确认nlu结果

## Agent
负责处理用户输入，调用slot filling模型和action模型。根据slot填写action 模板

调用tracker更新事件，调用database manger上传事件到数据库

## MemoryAgent
读取当前task故事模板，根据历史和当前intent判断机器回复模板标签，读取对应模板，生成回复。需要先训练nlu model

实例代码:当前任务为买电视  buy tv
```
manager = TrackerManager()
agent = MemoryAgent(manager)
t = Test(agent)
t.run()
```


## FormPolicy
设置若干槽位，顺序依次询问槽位值,填入用户输入（未加入ner及slotfilling）。

示例代码

```
# 设置所需信息槽位
p  = FormPolicy(["size","brand","resolution"])
print("User say:Hi")
# 询问第一个槽位
print("Bot say:{}".format(p.get_utter("Hi")))
while True:
    user_q = input("User say:")
    print("Bot say:{}".format(p.get_utter(user_q)))
```


## story流程定义文件
文件放置template文件夹中
