from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np
import os
import json

class split_dataset():
    def __init__(self, config, train, indices = None):
        self.logger = get_logger("Dataset_filter")
        # set log level to debug if requested
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        
        self.config = self.check_config(config)
        self.data_root = config["data_root"]
        if "~" in self.data_root:
            self.data_root = os.path.expanduser('~') + self.data_root[self.data_root.find("~")+1:]
        self.set_random_state()
        if indices == None:
            all_indices = [int(s[17:-5]) for s in os.listdir(self.data_root + "/parameters/")]
            # Split data into validation and training data 
            split = int(np.floor(config["validation_split"] * len(all_indices)))
            if self.config["shuffle_dataset"]:
                np.random.shuffle(all_indices)        
            # Load training or validation images as well as their indices
            if train:
                self.indices = all_indices[split:]
            else:
                self.indices = all_indices[:split]
        else:
            self.indices = indices

        print("len(sp.indices)",len(self.indices))
        self.indices = np.sort(self.indices)
        filter_dictionary = self.find_filters(train)
        self.indices = self.filter_dataset(filter_dictionary)
        print("len new_index:", len(self.indices))

    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
        else:
            self.config["random_seed"] = np.random.randint(0,2**32-1)
            np.random.seed(self.config["random_seed"])

    def check_config(self, config):
        assert "data_root" in config, "You have to specify the directory to the data in the config.yaml file."
        assert "filter" in config, "Can not filter without filters specified in config['filters']."
        if "same_filter" in config["filter"] and config["filter"]["same_filter"]:
            assert "train" in config["filter"] or "validation" in config["filter"], "You have to specify the filters in config['filter']['train'] or config['filter']['validation']"
        else:
            assert "train" in config["filter"], "You have to specify the filters for training in config['filter']['train']"
            assert "validation" in config["filter"], "You have to specify the filters for validation in config['filter']['validation']"
        return config

    def find_filters(self, train):
        """Find the filter keys for the training and valiadtion data set."""
        if "same_filter" in self.config["filter"] and self.config["filter"]["same_filter"]:
            filter_name = "train" if "train" in self.config["filter"] else "validation"
            return self.check_filter_keys(filter_name)
        else:
            if train:
                return self.check_filter_keys("train")
            else:
                return self.check_filter_keys("validation")

    def check_filter_keys(self, filter_name):
        """Returns a valid filter dictionary."""
        all_parameter_keys = ["total_branches", "total_cuboids", "scale", "same_scale", "same_theta", "theta", "phi", "same_material", "r", "g", "b", "a", "metallic", "smoothness", "DirectionalLightTheta", "DirectionalLightIntensity", 
        "CameraRes_width", "CameraRes_height", "Camera_FieldofView", "CameraRadius", "CameraTheta", "CameraPhi", "CameraVerticalOffset", "Camera_solid_background", 
        "totalPointLights", "PointLightsRadius", "PointLightsPhi", "PointLightsTheta", "PointLightsIntensity", "PointLightsRange", "PointLightsColor_r", "PointLightsColor_g", "PointLightsColor_b", "PointLightsColor_a", "same_PointLightsColor"   
        "totalSpotLights", "SpotLightsRadius", "SpotLightsPhi", "SpotLightsTheta", "SpotLightsIntensity", "SpotLightsRange", "SpotLightsColor_r", "SpotLightsColor_g", "SpotLightsColor_b", "SpotLightsColor_a", "same_SpotLightsColor"]
        filter_dict = self.config["filter"][filter_name]
        filter_keys = [ *self.config["filter"][filter_name] ] 
        for key in filter_keys:
            if key not in all_parameter_keys:
                del filter_dict[key]
                self.logger.info("Filter key not recognised:" + str(key) + "  This filter will be removed! Possible options are:  " + str(all_parameter_keys))
        return filter_dict

    def filter_dataset(self, filter_dict):
        
        def check_input_range(filter_range, assert_text = ""):
            """Check if filter range is valid type and length and convert ints."""
            if type(filter_range) in [int,float]:
                filter_range = [filter_range, filter_range]
            assert type(filter_range) == list, "filter has to be an int or a list with two elements." + assert_text
            assert len(filter_range) == 2,  "filter has to be an int or a list with two elements." + assert_text
            return filter_range
        
        def scalar_filter(value, filter_):
            if filter_[0] <= value <= filter_[1]:
                return True
            else:
                return False
        
        def list_filter(values, filter_):
            for value in values:    
                if not(filter_[0] <= value <= filter_[1]):
                    return False
            return True
        
        def _init_filter_base_bool(filter_dict, key):
            filter_dict_value = filter_dict[key]
            assert type(filter_dict_value) == bool, key + " filter has to be of type bool."
            def filtering(parameters):
                return filter_dict_value == parameters[key]
            return filtering

        def _init_filter_base_scalar(filter_dict, key):
            filter_dict_value = check_input_range(filter_dict[key])
            def filtering(parameters):
                return scalar_filter(parameters[key], filter_dict_value)
            return filtering
        
        def _init_filter_base_list(filter_dict, key):
            filter_dict_values = check_input_range(filter_dict[key])
            def filtering(parameters):
                return list_filter(parameters[key], filter_dict_values) 
            return filtering
        
        def _init_filter_total_branches(filter_total_branches):
            # take none also as a valid filter option 
            if filter_total_branches == None:
                filter_total_branches = [1,1]
            filter_total_branches = check_input_range(filter_total_branches, assert_text= "total_branches can also be None")
                
            def filtering_total_branches(parameters):
                if parameters["total_branches"] == None:
                    # create same structure for all parameters
                    test_total_branches = []
                    for i in range(parameters["total_cuboids"] - 1):
                        test_total_branches.append(1)
                    parameters["total_branches"] = test_total_branches
                # check all branch values
                return list_filter(parameters["total_branches"], filter_total_branches)
            
            # return the function to apply a fast filter 
            return filtering_total_branches
        
        
        filter_keys = [ *filter_dict ] 
        filter_functions = []
        if "total_branches" in filter_keys:
            filter_functions.append(_init_filter_total_branches(filter_dict["total_branches"]))
            self.logger.debug("total branches filter detected.")
        
        if "total_cuboids" in filter_keys:
            filter_functions.append(_init_filter_base_scalar(filter_dict, key="total_cuboids"))
            self.logger.debug("total cuboids filter detected.")
        if "scale" in filter_keys:
            filter_functions.append(_init_filter_base_scalar(filter_dict, key="scale"))
            self.logger.debug("scale filter detected.")
        
        if "same_scale" in filter_keys:
            filter_functions.append(_init_filter_base_bool(filter_dict, key = "same_scale"))
            self.logger.debug("same_scale filter detected.")
        if "same_theta" in filter_keys:
            filter_functions.append(_init_filter_base_bool(filter_dict, key = "same_theta"))
            self.logger.debug("same_theta filter detected.")
        if "same_material" in filter_keys:
            filter_functions.append(_init_filter_base_bool(filter_dict, key = "same_material"))
            self.logger.debug("same_material filter detected.")
        
        if "theta" in filter_keys:
            filter_functions.append(_init_filter_base_list(filter_dict, key = "theta"))
            self.logger.debug("same_material filter detected.")
        



        new_index = []
        for idx in self.indices:
            parameters = self.load_parameters(idx)
            for filter_function in filter_functions:
                filter_fits = filter_function(parameters)
                if not filter_fits:
                    break
            if filter_fits:
                new_index.append(idx) 

        return new_index
        
    def load_parameters(self, idx):
        # load a json file with all parameters which define the image 
        parameter_path = os.path.join(self.data_root, "parameters/parameters_index_" + str(idx) + ".json")
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters
    

import yaml

path = "/export/home/rhaecker/documents/research-of-latent-representation/GAN/configs/config_test.yaml"
with open(path) as f:
    config_ = yaml.safe_load(f)

sp = split_dataset(config = config_, train = True)
print("len(sp.indices)",len(sp.indices))
print("sp.indices", np.sort(sp.indices))
