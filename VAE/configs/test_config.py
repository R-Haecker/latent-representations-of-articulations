import yaml
import sys

sys.path.append('../')
from model.util import test_config


if len(sys.argv) > 1:
    config_path = str(sys.argv[1])
    if config_path[0]=="/":
        config_path = "." + config_path
    if config_path[-5:] != ".yaml":
        config_path = config_path + ".yaml"
    print("following config will be tested:", config_path)
else:
    config_path = "./config_test.yaml"
    print("No command line argument given checking config:", config_path)


with open(config_path) as fh:
    config = yaml.full_load(fh)    


test_config(config=config)
print("Config test was successful !")