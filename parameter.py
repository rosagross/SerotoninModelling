
class Parameter():

    def __init__(self):    
        
        # connectivity parameters
        self.Jee = 5
        self.Jei = 1
        self.Jie = 10
        self.Jii = 0.5 

        # the effective threshold for population activation  
        self.thetaE = 0
        self.thetaI = 0#25
        self.betaE = 0 
        self.betaI = 0

        # thetaE and betaE for Up-state regions
        self.thetaE_UP = 0 
        self.betaE_UP = 0
        self.Up_areas = [1, 3]

        # the 
        self.Eslope = 1
        self.Edesp = 5
        self.Islope = 4
        self.Idesp = 25 
        self.A = 1

        # noise (Ornstein-Uhlenbeck)
        #self.deltaGE = self.betaE
        
        # parameter determining strength of connectivity
        self.G=0
        # strength of serotonin stimulation
        self.S=0
        
        # time constants
        self.tauE = 10
        self.tauI = 2
        self.tauAdapt = 500
        self.tauN = 1
        self.sigmaN = 5 