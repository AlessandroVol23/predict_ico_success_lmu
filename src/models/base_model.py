from abc import ABC, abstractmethod 


class BaseModel(ABC):
    def get_name(self): 
        pass
    def get_params(self):
        pass
    def get_model(self, reinitialize = False):
       pass
    def fit(self,trn_x, trn_y, val_x=None, val_y=None, categorical_features = []):
        pass
    def predict_proba(self,oof_preds,sub_preds,X_test,folds,val_idx,val_x ):
        pass

