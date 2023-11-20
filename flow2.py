
from metaflow import FlowSpec, step, IncludeFile, current
from datetime import datetime
import os
from comet_ml import Experiment
import numpy as np
import os
import sys
from metaflow import Flow, Parameter
from metaflow import get_metadata, metadata

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from io import StringIO

assert os.environ.get('METAFLOW_DEFAULT_DATASTORE', 'local') == 'local'
assert os.environ.get('METAFLOW_DEFAULT_ENVIRONMENT', 'local') == 'local'

class Flow(FlowSpec):
    #input_train_file = IncludeFile('train.csv')
    #input_test_file = IncludeFile('test.csv')
    DATA_FILE_TRAIN = IncludeFile(
        'DATA_FILE_TRAIN',
        default='train.csv')
    DATA_FILE_TEST = IncludeFile(
        'DATA_FILE_TEST',
        default='test.csv')
    
    @step
    def start(self):
        
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        self.next(self.load_data)

    @step
    def load_data(self):

        self.train = pd.read_csv(StringIO(self.DATA_FILE_TRAIN))
        self.test = pd.read_csv(StringIO(self.DATA_FILE_TEST))
        print("Total {} rows in the training data.".format(len(self.train)))
        print("Total {} rows in testing data.".format(len(self.test)))
        
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        def preprocess(df):
            df = df.copy()
            
            def normalize_name(x):
                return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
            
            def ticket_number(x):
                return x.split(" ")[-1]
                
            def ticket_item(x):
                items = x.split(" ")
                if len(items) == 1:
                    return "NONE"
                return "_".join(items[0:-1])
            
            df["Name"] = df["Name"].apply(normalize_name)
            df["Ticket_number"] = df["Ticket"].apply(ticket_number)
            df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
            return df
        self.preprocessed_train_df = preprocess(self.train)
        self.preprocessed_test_df = preprocess(self.test)
        #self.preprocessed_train_df.drop(['Ticket'],axis = 1)
        self.validation = self.preprocessed_train_df.tail(100)
        self.serving = self.preprocessed_test_df
        self.train = self.preprocessed_train_df[:-100]

        self.input_features = list(self.preprocessed_train_df.columns)
        self.input_features.remove("Ticket")
        self.input_features.remove("PassengerId")
        self.input_features.remove("Survived")

        print(f"Input features: {self.input_features}")
        self.model_metric ={}
        self.hyperparams = [1,2]
        self.next(self.train_model, foreach='hyperparams')

    
    @step
    def train_model(self):
        self.hyperparam = self.input
        model = tfdf.keras.GradientBoostedTreesModel(
            verbose=0, # Very few logs
            features=[tfdf.keras.FeatureUsage(name=n) for n in self.input_features],
            exclude_non_specified_features=True, # Only use the features in "features"
            min_examples=self.hyperparam,
            categorical_algorithm="RANDOM",
            random_seed=1234,
        )
        X_train = tfdf.keras.pd_dataframe_to_tf_dataset(self.preprocessed_train_df, label="Survived")
        model.fit(X_train)

        self.evaluation = model.make_inspector().evaluation()
        self.metric = {"accuracy":self.evaluation.accuracy,"loss":self.evaluation.loss}
        print(f"Accuracy: {self.evaluation.accuracy} Loss:{self.evaluation.loss}")

        self.model_name = f"model_min_examples_{self.hyperparam}.tf"
        self.next(self.join)

    @step
    def join(self,inputs):
        exp = Experiment(
            api_key="Qz6bevwp9dAId5GMMOZhiUnW2",
            project_name="hd2367nyu_comet_test",
            workspace = 'duhanlu'
        )

        self.best_val_accuracy = max(inp.evaluation.accuracy for inp in inputs)
        best_models = [inp for inp in inputs if inp.metric['accuracy'] == self.best_val_accuracy]
        if best_models:
            best_model = best_models[0]
            self.best_model_name = best_model.name
        exp.log_metrics({'best_val_accuracy': self.best_val_accuracy,'loss': best_model.evaluation.loss})
        exp.log_parameters(best_model.hyperparam)
        #exp.log_metrics({'loss': best_acc})
       
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))

    
if __name__ == '__main__':
    Flow()






