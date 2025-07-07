from abc import ABC, abstractmethod
import numpy as np

class VehicleModel(ABC):
    '''
    Class information
    - State: [x position, y position, theta heading, x velocity, y velocity, angular velocity]
        - Position is relative to cost map origin
    - Cost Map: Nav2 CostMap
    '''

    def __init__(self, cost_map, initial_state=None):
        self.current_state = np.array(initial_state) if initial_state is not None else np.zeros(4)
        self.cost_map = cost_map

    @property
    def state(self):
        return self.current_state
    
    @property
    def cost_map(self):
        return self.cost_map

    @abstractmethod
    def predict(self, state, control_input, dt):
        """Predict the next state given the current state, control input, and timestep."""
        pass

    @abstractmethod
    def reset(self, new_state, cost_map = None):
        """Reset any internal states (for RNNs, etc.)."""
        pass



class EkfVehicleModel(VehicleModel):
    def __init__(self, cost_map, initial_state=None):
        super().__init__()
        # Init EKF parameters here
        pass

    def predict(self, state, control_input, dt):
        # Perform EKF prediction here
        next_state = state + control_input * dt  # Example linear update
        return next_state

    def reset(self, new_state, new_cost_map = None):
        self.current_state = np.array(new_state)

        if new_cost_map:
            self.cost_map = new_cost_map



class RNNVehicleModel(VehicleModel):
    def __init__(self, cost_map, initial_state=None):
        super().__init__()
        pass

    def predict(self, state, control_input, dt):
        pass

    def reset(self, new_state, new_cost_map = None):
        pass