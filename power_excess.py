from TGAS.mcmc_class import MCMC
import numpy as np

class Scaling(MCMC):                                                                                      
                                                                                                          
    def __init__(self, debug=False):                                                                                                                                                                                                        
        super(MCMC, self).__init__()
        self.problem = 'single'
        self.nwalkers = 20                                                                               
        self.nsteps = 30000 #3000                                                                              
        self.rejection_fraction = 0.8                                                                    
        self.burninsteps = 10000 #1000                                                                           
        self.debug = debug  
        # Gio: set this to how many threads you have on your computer minus a couple so they're not all taken up by computation...
        self.threads = 8
        
        
        # Gio: read in the frequency and amp_tot arrays here:
        self.freq = f_tot #np.linspace(0,100,10000)
        self.power = p_tot#Amp_tot
        
        if self.debug:
            self.freq = np.linspace(0,100,10000)
            self.power = 100.0*np.exp(-(self.freq - 28.0)**2/2.0/(10.0)**2)/np.sqrt(2.0*np.pi*10.0**2) + 0.1
      
    @property                                                                                             
    def guess(self):                                                                                      
        if self.problem == 'single':   
            self._guess = np.log(np.array([2000.0, 1.0, 10.0, 0.01])  ) 
            self.names = np.array(['numax', 'width', 'amp', 'noise'])  
        return self._guess                                                                                
                                                                                                          
    @property                                                                                             
    def nvariables(self):                                                                                 
        if self.problem == 'single':                                                                      
            self._nvariables = 4                                                                                                                                                
        return self._nvariables
    
    @nvariables.setter                                                                                    
    def nvariables(self, value):                                                                          
        self._nvariables = value                                                                          
    
    def lnprior(self, *args):
        if np.exp(args[0]) > 5000:
            return -np.inf
        if np.exp(args[0]) < 0.01:
            return -np.inf
        if np.exp(args[2]) < 0.0001: #change for amplitude 
            return -np.inf
        if np.exp(args[2]) > 1e8:
            return -np.inf
        if np.exp(args[1]) > 5000: #width of distribution and add args1 < #
            return -np.inf
        if np.exp(args[1]) < 0.000005: #add or remove to best fit 
            return -np.inf
        if np.exp(args[3]) > 1e8:
            return -np.inf
        if np.isinf(np.abs(np.array(args))).any():
            return -np.inf
        return 0.0
    
    def setup_data(self):
        pass
    
    def model(self, *args):
        return np.exp(args[2])*np.exp(-(np.exp(args[0])-self.freq)**2/2.0/(np.exp(args[1]))**2)/np.sqrt(2*np.pi*np.exp(args[1])**2) + np.exp(args[3])
    
    def logl(self, *args):  
        mod = self.model(*args[0])
        
        if np.isinf(np.abs(mod)).any():
            return -np.inf
        
        logl = -np.sum(np.log(mod)) - np.sum(self.power/mod) + self.lnprior(*args[0])
            
        if np.isnan(logl):                                                                                
            return -np.inf                                                                               
        
        return logl
             
def main():                                                                                               
    scaling = Scaling(debug=False)                                                                         
    scaling.run()    
    print(scaling.model(*scaling.best_fit))
    print(scaling.power)
    print(scaling.freq)
    params = scaling.samples                                                                              
    params_best = scaling.best_fit                                                                        
    print('best-fitting numax : {:9.3f}uHz +/- {:9.3f}uHz'.format(np.exp(params_best[0]), np.exp(params_best[0])*scaling.error[0]))                          
    print('best-fitting width : {:9.3f}uHz +/- {:9.3f}uHz'.format(np.exp(params_best[1]), np.exp(params_best[1])*scaling.error[1]))                                                  
    print('best-fitting amp : {:9.3f}ppm^2/uHz +/- {:9.3f}ppm^2/uHz'.format(np.exp(params_best[2]), np.exp(params_best[2])*scaling.error[2]))                          
    print('best-fitting noise : {:9.3f}ppm^2/uHz +/- {:9.3f}ppm^2/uHz'.format(np.exp(params_best[3]), np.exp(params_best[3])*scaling.error[3]))                                                                                                           
