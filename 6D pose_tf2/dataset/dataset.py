import json
import os

data_path = "scene_gt.json"

path = "E:/proj/Project tweets/2019_07_01/01/"
filelist = os.listdir(path)

for filename in filelist:
    if filename.endswith(".json"):
        with open(path + filename, 'r') as file:
            lines = file.readlines()
            for line in lines:
                json_line = json.loads(line)
                try:
                    tags = ["created_at", "text", "user_location"]
                    for i in json_line:
                        if i not in tags:
                            del json_line[i]
                    with open('out_01_01.json', 'w') as file:
                        file.write(json.dumps(json_line))
                except ValueError:
                    continue



# Opening JSON file
#f = open(data_path, )

# returns JSON object as
# a dictionary
# data = json.load(f)

# Iterating through the json
# list
# for i in data['0']:
#     if i['obj_id'] != 1:
#         i.clear()
#     with open('out.json','a') as f:
#         # json.dump(i.)
#         json.dump(i, f)
#         print(i)





# for index in range(1000):
#     for i in data[str(index)]:
#         if i['obj_id'] != 1:
#             i.clear()
#             # del i['obj_id']
#             # del i['cam_R_m2c']
#             # del i['cam_t_m2c']
#         with open('out.json','w') as f:
#             f.write(json.dumps())
#             json.dump(i, f)
#         print(i)



# Closing file
f.close()