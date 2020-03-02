import yaml
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

with open("plot_var_upsample.yaml") as fh:
    config = yaml.full_load(fh)    

runs_path = []

for s in os.listdir(config["run_path"]):

    #print(s[-len(config["run_name"]) : -1])
    if s[-len(config["run_name"])-1:-1] == config["run_name"]:
        print("found:", s)
        runs_path.append(s)

for s in runs_path:
    conf = s + "/configs/"
    config_name = os.listdir(config["run_path"] + conf)
    with open(config["run_path"] + conf + config_name[-1]) as fh:
        test_conf = yaml.full_load(fh)
    if "explanation" in config:
        if config["explanation"] != test_conf["explanation"]:
            runs_path.remove(s)

sorted_runs = np.sort(runs_path)
final_runs_path = []
for i in range(config["amount_of_runs"]):
    final_runs_path.append(config["run_path"] + sorted_runs[-i] + "/train/validation/log_op/")


runs_all_img = []
step_strings = []
for s in final_runs_path:
    print(s)
    run_img = []
    images_path = os.listdir(s)
    step_strings.append(images_path[:][-11:-4])
    for img_path in images_path:
        if config["inputs"]==False:
            #print(img_path[:6])
            if img_path[:6]!="inputs":
                run_img.append((Image.open(s + img_path)).crop((0,0,64,64)))
    runs_all_img.append(run_img)

print(step_strings)

run_img = []
for i in range(len(runs_all_img)-1):
    img = []

    for images in runs_all_img[i]:
    #for j in range(36):
        #img.append(np.array(runs_all_img[i][j]))
        img.append(np.array(images))
    print("finished run len:",len(img))
    run_img.append(np.vstack(img))
final_image = np.hstack(run_img)

plt.figure(figsize=(10,30))
plt.imshow(final_image)
plt.show()
