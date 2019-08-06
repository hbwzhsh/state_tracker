import os
#task_config_path = "C://Users//Administrator//PycharmProjects//state_starker//config//config.cf"
file_path = os.path.join(os.getcwd(),"config")
file_path = os.path.join(file_path,"config.cf")
task_config_path = file_path

#print(file_path)
def load_config(file_path,delimiter = "="):
    file  =open(file_path,"r")
    d = {}
    for row in file.readlines():
        row = row.strip("\n")
        k,v = row.split(delimiter)
        d[k] = v
    return d

def get_task_name():
    d = load_config(task_config_path,delimiter=":")
    return d.get("task")

def type_set(types_d,config_d):
    for k,v in types_d.items():
        config_d[k] = v(config_d)
    return config_d

def auto_type_set(config_d):
    for k,v in config_d.items():
        if v.isdigit():
            if "." in v:
                config_d[k] = float(v)
            else:
                config_d[k] = int(v)
    return config_d

def read_temp(path,delimiter=":"):
    file = open(path, "r",encoding="utf-8")
    d = {}
    for row in file.readlines():
        row = row.strip("\n").strip(" ")
        if delimiter not in row:
            continue
        if len(row.split(delimiter))!=2:
            print("invalid row",row)
            continue
        k,v = row.split(delimiter)
        k = k.strip("\n").strip(" ")
        v = v.strip("\n").strip(" ")
        d[k] = v
    file.close()
    return d

