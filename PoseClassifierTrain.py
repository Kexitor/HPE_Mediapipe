from sklearn.neural_network import MLPClassifier
from MediapipeUtilities import MediapipeUtilities
import pickle


class PoseClassifierTrain:
    def __init__(self):
        self.train_coordinates = None
        self.train_poses_num = None
        self.trained_model = None
        self.trained_model_save_path = None

    def init_train_data(self, train_file_folder: str, train_file_name: str):
        """
        Initializes and prepares data for training

        :param train_file_folder: file directory/folder without file name
        :param train_file_name: file name
        """
        train_poses, self.train_coordinates = MediapipeUtilities().csv_converter(train_file_folder, train_file_name)
        self.train_poses_num = MediapipeUtilities().pose_to_num(train_poses)

    def train_classifier(self,
                         solver='lbfgs',
                         activation='logistic',
                         alpha=1e-5,
                         hidden_layer_sizes=(150, 10),
                         random_state=1,
                         max_iter=10000):
        """
        Trains MLPClassifier pose classifier

        :param solver: {'lbfgs', 'sgd', 'adam'}, default='adam' The solver for weight optimization.
        :param activation:  {'identity', 'logistic', 'tanh', 'relu'}, default='relu' Activation function for the hidden layer.
        :param alpha: Strength of the L2 regularization term. The L2 regularization term is divided by the sample size when added to the loss.
        :param hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
        :param random_state: int, RandomState instance, default=None Determines random number generation for weights and bias initialization, train-test split if early stopping is used, and batch sampling when solver='sgd' or 'adam'.
        :param max_iter: Maximum number of iterations. The solver iterates until convergence (determined by 'tol') or this number of iterations. For stochastic solvers ('sgd', 'adam'), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.
        :return: trained model
        """
        model = MLPClassifier(solver=solver, activation=activation, alpha=alpha, hidden_layer_sizes=hidden_layer_sizes,
                              random_state=random_state, max_iter=max_iter)
        trained_model = model.fit(self.train_coordinates, self.train_poses_num)
        self.trained_model = trained_model

        return trained_model

    def save_model(self, save_path: str, trained_model=None):
        """
        Saves pretrained model

        :param trained_model: model if non None
        :param save_path: model save path
        :return: str with full model save path with extension ".pkl"
        """

        full_save_path = save_path + ".pkl"
        self.trained_model_save_path = full_save_path
        if trained_model:
            with open(full_save_path, 'wb') as file:
                pickle.dump(trained_model, file)
        else:
            with open(full_save_path, 'wb') as file:
                pickle.dump(self.trained_model, file)

        return full_save_path
