import os

class Paths:
    def __init__(self, path_to_project, model):
        '''
        Defines the directory path structure given a model, and the root project path.
        Variables:
            path_to_project: str
                The path from home directory to the root of the project.
            
            model: str
                The shortname of the model. For this case either "alon" or "bhalla"
        
        Returns:
            This function returns an object with attributes of all the relevant file
            directories for the biophys_glm project.
        '''
        # set model attribute
        self.model = model
        
        # set data directories
        self.home = os.path.expanduser("~")
        self.project = os.path.join(self.home, path_to_project)
        self.data = os.path.join(self.project, 'data', model)
        self.slopes = os.path.join(self.data, 'beta_slopes')
        self.example_slopes = os.path.join(self.data, 'example_slopes_kA')
        self.bases = os.path.join(self.data, 'glm_basis')
        self.glm_sims = os.path.join(self.data, 'glm_simulated_spikes')
        self.survival = os.path.join(self.data, 'glm_survival')
        self.reconstructed = os.path.join(self.data, 'reconstructed_stimulus')
        self.coherence = os.path.join(self.data, 'coherence')
        self.sim_slopes = os.path.join(self.data, 'example_simulation_kca3')
        self.biophys_test_in = os.path.join(self.data, 'biophys_test_in')
        self.biophys_test_out = os.path.join(self.data, 'biophys_test_out')
        self.biophys_input = os.path.join(self.data, 'biophys_input')
        self.biophys_output = os.path.join(self.data, 'biophys_output')

        # set analysis directories
        self.figures = os.path.join(self.project, 'analysis', 'figures')
        self.tables = os.path.join(self.project, 'analysis', 'tables')
        self.model_figures = os.path.join(self.project, 'analysis', 'figures', model)
        self.model_tables = os.path.join(self.project, 'analysis', 'tables', model)
        self.manuscript = os.path.join(self.project, 'manuscript')


    def __repr__(self):
        return 'Object with directory paths for the {} model.'.format(self.model)