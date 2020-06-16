from edflow import get_logger
from edflow.custom_logging import LogSingleton
import numpy as np
import os
import json

#TODO implemet filtering paarmeters save lokaly and save filter local and in dataset 
class filter_dataset():
    def __init__(self, config, train):
        if "debug_log_level" in config and config["debug_log_level"]:
            LogSingleton.set_log_level("debug")
        self.logger = get_logger("Dataset_filter")
        self.data_root = self.get_data_root(config)
        self.config = self.check_config(config)
        
        self.set_random_state()
        self.all_indices = self.load_all_indices()
        # is a filter specified
        if "filter" in self.config:
            # is the filter the same for validation and training dataset
            if "same_filter" in self.config["filter"] and self.config["filter"]["same_filter"]:
                # Filter the indices
                indices_filtered = self.get_filtered_indices()
                self.logger.debug("Length of filtered dataset: " + str(len(indices_filtered)) + " / " + str(len(self.all_indices)))
                # Split the indices according to train
                self.indices = self.split_indices(indices_filtered, train)
            else:
                # check if the filters are already saved
                all_filter_dict = self.get_all_filter()
                filter_path = self.search_saved_filters(all_filter_dict)
                if filter_path != None:
                    # load saved indices for the filter
                    filter_path = filter_path + "_train" if train else filter_path + "_validation" 
                    self.indices = self.load_indices(filter_path)
                    if train and "shuffle_train" in self.config and self.config["shuffle_train"]:
                        np.random.shuffle(self.indices)
                else:
                    # filter the data set 
                    train_indices_filtered, val_indices_filtered = self.create_new_filters()
                    self.save_filter(all_filter_dict, train_indices_filtered, val_indices_filtered)
                    self.indices = train_indices_filtered if train else val_indices_filtered
                    if self.config["shuffle_dataset"]:
                        np.random.shuffle(self.indices)
                    if train and "shuffle_train" in self.config and self.config["shuffle_train"]:
                        np.random.shuffle(self.indices)
        else:
            # no filter: load and split indices normaly
            self.indices = self.split_indices(self.all_indices, train)
            self.logger.debug("No data set filter detected.")
        self.logger.info(" train: " + str(train) + ", data set length " + str(len(self.indices)))
        self.logger.debug(" first ten indices of dataset: " + str(self.indices[:10]))

    def get_data_root(self, config):
        # Get the directory to the data
        assert "data_root" in config, "You have to specify the directory to the data in the config.yaml file."
        data_root = config["data_root"]
        if "filter" in config and "local" in config["filter"] and config["filter"]["local"]:
            data_root = "/export/home/rhaecker/documents/research-of-latent-representation/data_parameters" + data_root[data_root.rfind("/"):]
        else:
            if "~" in data_root:
                data_root = os.path.expanduser('~') + data_root[data_root.find("~")+1:]
        self.logger.debug("data_root: " + str(data_root))
        return data_root

    def create_new_filters(self):
        # train indices
        train_indices_filtered = self.get_filtered_indices(train = True)
        # validation indices
        val_indices_filtered = self.get_filtered_indices(train = False)
        # get all intersection elements between validaiton and tarining data set
        intersection_indices = [x for x in train_indices_filtered if x in val_indices_filtered]
        if self.config["shuffle_dataset"]:
            np.random.shuffle(intersection_indices)
        self.logger.debug("Length of intersection data set = " + str(len(intersection_indices)) + " / " + str(len(train_indices_filtered) + len(val_indices_filtered)))
        def tp_():
            '''Compare sizes of filtered dataset in percent.'''
            return len(train_indices_filtered)/(len(train_indices_filtered) + len(val_indices_filtered))
        # remove duplicated elements from data set wich should have fewer elements according to the split
        tp = tp_()
        while len(intersection_indices) > 0:
            if tp > (1 - self.config["validation_split"]):
                train_indices_filtered.remove(intersection_indices.pop(0))
            else:
                val_indices_filtered.remove(intersection_indices.pop(0))
            tp = tp_()
        self.logger.debug("Removed intersections. Splitting filtered data set completed.")
        self.logger.debug("New length of train data set = " + str(len(train_indices_filtered)))
        self.logger.debug("New length of validation data set = " + str(len(val_indices_filtered)))
        self.logger.info("Given split = " + str(self.config["validation_split"]) + " , actual split = " + str(tp))
        self.logger.info("New length of data set = " + str(len(val_indices_filtered) + len(train_indices_filtered)) + " / " + str(len(self.all_indices)))

        return train_indices_filtered, val_indices_filtered
        
    def get_all_filter(self):
        # create finished filter dict
        all_filter_dict = {}
        all_filter_dict["train"] = self.find_filters(train=True)
        all_filter_dict["validation"] = self.find_filters(train=False)
        all_filter_dict["validation_split"] = self.config["validation_split"]
        return all_filter_dict

    def get_filtered_indices(self, train = None):
        filter_dict = self.find_filters(train)
        if filter_dict == None:
            # no filter
            indices_filtered = self.split_indices(self.all_indices, train)
            print(" WORKED HERE")
        else:
            filter_dict_path = self.search_saved_filters(filter_dict)
            if filter_dict_path != None:
                indices_filtered = self.load_indices(filter_dict_path)
                self.logger.debug("Saved filter found and loaded.")
            else:
                indices_filtered = self.filter_entire_dataset(filter_dict)
                self.save_filter(filter_dict, indices_filtered)
                self.logger.debug("Filter applied to whole data set.")
            self.logger.debug("Length of filtered data set = " + str(len(indices_filtered)) + " / " + str(len(self.all_indices)))
        return indices_filtered

    def load_all_indices(self):
        # Load every indices from all images
        if "request_tri" in self.config and self.config["request_tri"]:
            every_indices = [int(s[17:-5]) for s in os.listdir(self.data_root + "/parameters/")]
            all_indices = []
            for i in range(int(np.floor(len(every_indices)/3))):
                all_indices.append(every_indices[i*3])
        else:
            all_indices = [int(s[17:-5]) for s in os.listdir(self.data_root + "/parameters/")]
        return np.sort(all_indices)    

    def split_indices(self, input_indices, train):
        # Split data into validation and training data 
        if self.config["shuffle_dataset"]:
            np.random.shuffle(input_indices)
        else:
            input_indices = np.sort(input_indices)
        split = int(np.floor(self.config["validation_split"] * len(input_indices)))
        indices = input_indices[split:] if train else input_indices[:split]
        if train and "shuffle_train" in self.config and self.config["shuffle_train"]:
            np.random.shuffle(indices)
        return indices

    def set_random_state(self):
        if "random_seed" in self.config:
            np.random.seed(self.config["random_seed"])
        else:
            self.config["random_seed"] = np.random.randint(0,2**32-1)
            np.random.seed(self.config["random_seed"])

    def load_parameters(self, idx):
        # load a json file with all parameters which define the image 
        parameter_path = os.path.join(self.data_root, "parameters/parameters_index_" + str(idx) + ".json")
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters
    
    def check_config(self, config):
        assert "data_root" in config, "You have to specify the directory to the data in the config.yaml file."
        if "filter" in config:
            if "same_filter" in config["filter"] and config["filter"]["same_filter"]:
                assert bool("train" in config["filter"]) != bool("validation" in config["filter"]), "You have to specify the filters in config['filter']['train'] or config['filter']['validation']. Only specify one if 'same_filter' is true."
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
        
        self.all_parameter_bool_keys = ["same_scale", "same_theta", "same_material", "Camera_solid_background", "same_PointLightsColor", "same_SpotLightsColor"]
        
        self.all_parameter_scalar_keys = ["total_cuboids", "phi", "DirectionalLightTheta", "DirectionalLightIntensity", 
                                    "CameraRes_width", "CameraRes_height", "Camera_FieldofView", "CameraRadius", "CameraTheta", "CameraPhi", "CameraVerticalOffset", 
                                    "totalPointLights", "totalSpotLights"]

        self.all_parameter_list_keys = ["theta", 
                                    "PointLightsRadius", "PointLightsPhi", "PointLightsTheta", "PointLightsIntensity", "PointLightsRange", "PointLightsColor_r", "PointLightsColor_g", "PointLightsColor_b", "PointLightsColor_a",
                                    "SpotLightsRadius", "SpotLightsPhi", "SpotLightsTheta", "SpotLightsIntensity", "SpotLightsRange", "SpotLightsColor_r", "SpotLightsColor_g", "SpotLightsColor_b", "SpotLightsColor_a"]

        self.parameter_material_keys = ["r", "g", "b", "a", "metallic", "smoothness"]

        filter_dict = self.config["filter"][filter_name]
        if filter_dict != None:
            filter_keys = [ *self.config["filter"][filter_name] ] 
            for key in filter_keys:
                if key not in all_parameter_keys:
                    del filter_dict[key]
                    self.logger.info(" Filter key not recognised:" + str(key) + "  This filter will be removed! Possible options are:  " + str(all_parameter_keys))
        return filter_dict

    def filter_entire_dataset(self, filter_dict):
        
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
            if values == None:
                return False
            else:
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
        
        def _init_filter_base_scalar_or_list(filter_dict, key, scalar_key):
            filter_dict_values = check_input_range(filter_dict[key])
            def filtering(parameters):        
                if parameters[scalar_key]:
                    return scalar_filter(parameters[key], filter_dict_values) 
                else:
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
        for filter_key in filter_keys:
            if filter_key in self.all_parameter_bool_keys:
                filter_functions.append(_init_filter_base_bool(filter_dict, key = filter_key))
                self.logger.debug( filter_key + " filter detected.")
            elif filter_key in self.all_parameter_scalar_keys:
                filter_functions.append(_init_filter_base_scalar(filter_dict, key = filter_key))
                self.logger.debug( filter_key + " filter detected.")
            elif filter_key in self.all_parameter_list_keys:
                filter_functions.append(_init_filter_base_list(filter_dict, key = filter_key))
                self.logger.debug( filter_key + " filter detected.")
            elif filter_key in self.parameter_material_keys:
                filter_functions.append(_init_filter_base_scalar_or_list(filter_dict, key = filter_key, scalar_key="same_material"))
                self.logger.debug( filter_key + " filter detected.")
            elif filter_key == "scale":
                filter_functions.append(_init_filter_base_scalar_or_list(filter_dict, key = filter_key, scalar_key="same_scale"))
                self.logger.debug( filter_key + " filter detected.")
            elif filter_key == "total_branches":
                filter_functions.append(_init_filter_total_branches(filter_dict["total_branches"]))
                self.logger.debug("total_branches filter detected.")
            else:
                assert 1==0, filter_key + " this key should not exists filter not complete or filterkeys not checked right"

        new_index = []
        i = 0
        for idx in self.all_indices:
            i += 1
            self.printProgressBar(i, len(self.all_indices), prefix = 'Filtering:', suffix = 'Complete', length = 100)
            parameters = self.load_parameters(idx)
            for filter_function in filter_functions:
                filter_fits = filter_function(parameters)
                if not filter_fits:
                    break
            if filter_fits:
                new_index.append(idx) 

        return new_index
    
    def save_filter(self, filter_dict, indices_train, indices_validation = None):
        path = self.data_root + "/filter"
        index = 0
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            for el in os.listdir(path):
                if el[:-5] == ".json":
                    index += 1
        path = path + "/filter_" + str(index)
        with open(path + ".json", 'w') as f:
            json.dump(filter_dict, f)
            f.close()
        if indices_validation == None:
            np.save(path, indices_train)
        else:
            np.save(path + "_train", indices_train)
            np.save(path + "_validation", indices_validation)

    def load_indices(self, path):
        if path[-5:] == ".json":
            path = path[:-5] + ".npy"
        else:
            path = path + ".npy"
        self.logger.debug(" Loading indices from path: " + str(path))
        return np.load(path)

    def search_saved_filters(self, filter_dictionary):
        self.logger.debug("searching for filter: " + str(filter_dictionary))
        path = self.data_root + "/filter"
        if os.path.exists(path):
            filters = os.listdir(path)
            for filter_name in filters:
                if filter_name[-5:] == ".json":
                    path = path + "/" + filter_name
                    with open(path, "r") as f:
                        testing_filter = json.load(f)
                    self.logger.debug("testing filter: " + str(filter_name) + " = " + str(testing_filter))
                    # check if dictionaries are exactly the same
                    if testing_filter == filter_dictionary:
                        self.logger.debug("equal testing filter: " + str(filter_name) + " = " + str(testing_filter))
                        return path[:-5]
        return None

    def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()

'''import yaml

path = "/export/home/rhaecker/documents/research-of-latent-representation/GAN/configs/config_test.yaml"
with open(path) as f:
    config_ = yaml.safe_load(f)

sp = filter_dataset(config = config_, train = True)
'''