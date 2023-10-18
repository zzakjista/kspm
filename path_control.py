from path import Path
from config import ROOT_DATA_PATH

class path_master(Path):

    def __init__(self):
        self.rawdata_path = 'kospi200_10Y_price_investor.pickle'
        self.preprocessed_data_path = 'preprocessed_data.pickle'


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

    def get_strategic_data_folder_path(self):
        root_data_path = self.get_data_root_path()
        strategic_data_folder_path = root_data_path.joinpath('strategic_data')
        return strategic_data_folder_path
    
    

