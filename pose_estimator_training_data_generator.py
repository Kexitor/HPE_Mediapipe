from TrainingDataGenerator import TrainingDataGenerator
from data_lists import train_data, test_data

# train_data and test_data are made by me, if you use other videos for training, you will need make your own markup

path_to_training_files = "videos/cuts_test/"
csv_save_path = "csvs/test1"

train_data_generator = TrainingDataGenerator()
train_data_generator.init_main_params(test_data, csv_save_path, path_to_training_files)
train_data_generator.generate_train_csv()
