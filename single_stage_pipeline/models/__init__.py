from .clam_base import CLAM, CLAMAttention, CLAMGatedAttention
from .clam_regressor import CLAMRegressor, create_clam_regressor
from .clam_classifier import CLAMClassifier, create_clam_classifier, OrdinalLoss

__all__ = [
    'CLAM', 'CLAMAttention', 'CLAMGatedAttention',
    'CLAMRegressor', 'create_clam_regressor', 
    'CLAMClassifier', 'create_clam_classifier', 'OrdinalLoss'
] 