r"""
 _____ ____  ____   ___  ____  ____  
| ____|  _ \|  _ \ / _ \|  _ \/ ___| 
|  _| | |_) | |_) | | | | |_) \___ \ 
| |___|  _ <|  _ <| |_| |  _ < ___) |
|_____|_| \_\_| \_\\___/|_| \_\____/ 
                                     
"""

import warnings

########## DATA FETCH

class FetchedDataWarning(UserWarning):
    pass

class FetchedDataError(Exception):
    pass


########## IV SOLVER 

class VolSolverWarning(UserWarning):
    pass

class VolSolverError(Exception):
    pass

############## SVI

class SVIParamError(Exception):
    pass

class SVIFitWarning(UserWarning):
    pass

class SVIFitError(Exception):
    pass