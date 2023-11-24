from path import Path
from config import *

class path_master(Path):

    def __init__(self, args):
        self.rawdata_path = args.rawdata_path
        self.preprocessed_data_path = args.preprocessed_data_path

    def get_data_root_path(self):
        return Path(ROOT_DATA_PATH)
    
    def get_rawdata_folder_path(self):
        root_data_path = self.get_data_root_path()
        rawdata_folder_path = root_data_path.joinpath('rawdata')
        return rawdata_folder_path
    
    def get_preprocessed_folder_path(self):
        root_data_path = self.get_data_root_path()
        preprocessed_folder_path = root_data_path.joinpath('preprocessed')
        return preprocessed_folder_path
    
    def get_scaler_root_path(self):
        return Path(ROOT_SCALER_PATH)
    
    def get_pretrained_root_path(self):
        return Path(ROOT_EXPERIMENT_PATH)
    
    def get_pretrained_folder_path(self, args):
        pretrained_folder_path = self.get_pretrained_root_path().joinpath(args.version).joinpath(args.pretrained_version)
        print(pretrained_folder_path)
        if not pretrained_folder_path.exists():
            raise Exception('pretrained folder does not exist')
        return pretrained_folder_path

    