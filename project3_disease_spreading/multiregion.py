from model import model
import numpy as np

class multiregion_model(model):
    """
    Object for representing SIR model with multiple regions.
    Input should be a list of different model classes
    """
    def __init__(self, regions: list[model], model_function: str):

        self.regions = regions
        self.n_regions = len(regions)

        if self.n_regions < 2:
            raise ValueError("Provide more than one \"model\" class for multiregion class")
        
        # Combine regional parameters
        self.total_parameters = {
            key: np.array([
                region.total_parameters[key]
                for region in regions
            ])
            for key in regions[0].total_parameters
        }

        # Select initial status
        self.total_parameters["initial_status"] = np.concatenate(self.total_parameters["initial_status"],axis=0)
        self._initial_status = self.total_parameters["initial_status"]
        
        # Select multiregional model function
        self._set_model_function(model_function)