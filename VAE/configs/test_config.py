import yaml
import sys

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

sys.path.append('../')
from model.vae import test_config_model

# Test for model parameters
test_config_model(config=config)

# Test config for iterator parameters
assert "loss_function" in config, "The config must contain and define a Loss function. possibilities:{'L1','L2'or'MSE','KL'or'KLD'}."
assert "learning_rate" in config, "The config must contain and define a the learning rate."
assert "weight_decay" in config, "The config must contain and define a the weight decay."

print("Test was successful !")