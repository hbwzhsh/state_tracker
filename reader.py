def load_config(file_path,delimiter = "="):
    file  =open("config.cf","r")
    d = {}
    for row in file.readlines():
        row = row.strip("\n")
        k,v = row.split(delimiter)
        d[k] = v
    return d

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
        k,v = row.split(delimiter)
        k = k.strip("\n").strip(" ")
        v = v.strip("\n").strip(" ")
        d[k] = v
    file.close()
    return d

