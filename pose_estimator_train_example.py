from PoseClassifierTrain import PoseClassifierTrain

train_dataset_directory = "csvs/"
train_dataset_filename = "37vtrain_mp.csv"
save_file_path = "pretrained_models/test1"

classifier_trainer = PoseClassifierTrain()
classifier_trainer.init_train_data(train_dataset_directory, train_dataset_filename)
trained_model = classifier_trainer.train_classifier()  # params could be modified
saved_model_path = classifier_trainer.save_model(save_file_path, trained_model)
print("Training process can take some time, depending on CPU power")
print(saved_model_path)
