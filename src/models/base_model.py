from abc import ABC, abstractmethod 


class BaseModel(ABC):
    def get_name(self): 
        pass
    def get_params(self):
        pass
    def get_model(self, reinitialize = False):
       pass
    def fit(self,trn_x, trn_y, val_x, val_y, categorical_features = []):
        pass
    def predict_proba(self,oof_preds,sub_preds,X_test,folds,val_idx,val_x ):
        pass
    def get_feature_importance(self, X_test, y_test, categorical_features_indices=[]):
        pass
