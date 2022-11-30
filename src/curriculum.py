import train
from util import best_generalization_agent
class Curriculum:
    def __init__(self, training_schedule, training_parameters, best_criteria = best_generalization_agent):
        self.training_schedule = training_schedule
        self.training_parameters = training_parameters
        self.trained_schedule_best_model_paths = [] #lista de paths a los modelos usados
    def trainSchedule(self):
        """
        idea
        si training_schedule[0] = (2,2)
            train_agent([(problem, 2, 2)], file, features, optimizer="sgd", model="pytorch",
                           first_epsilon=1, last_epsilon=0.01, epsilon_decay_steps=250000, early_stopping=True,
                          copy_freq=5000, total_steps=5000000 * 2)
            usar best_generalization_agent((problem, 2, 2)], file) como base
            para train_agent([(problem, 2, 3)] si training_schedule[1]=(2,3)

        COMO LEVANTO LOS MODELOS DE ONNX A PYTORCH O SKLEARN?
        OPCION 1> HACER UN PROGRAMA A MANO QUE PEUDA HACER LA CONVERSION
        OPCION 2> NO USAR ONNX O CONVERTIR ONNX A UN FORMATO INTERMEDIO
        """
        pass

