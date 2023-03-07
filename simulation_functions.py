
import numpy as np 
import matplotlib.pyplot as plt
import math
import pandas as pd
class SimulationSession():

    def __init__(self, par, sim_params, output_dir):
        # a session needs parameters and output functions 
        self.par = par
        self.nrVars = 3 # E, I and A
        self.nrAreas = 14
        self.sim_params = sim_params
        self.nrSteps = int((sim_params[1]-sim_params[0])/sim_params[2])
        self.initial_cond = np.zeros((self.nrVars, self.nrAreas))
        self.noise_init = np.zeros((2, self.nrAreas))
        self.output_dir = output_dir
        self.rate = np.zeros((self.nrVars, self.nrSteps, self.nrAreas))

        # the random values for the noise are generated now already
        # TODO: check if the the noise is correct (with the mu and sigma)
        self.random_vals = np.random.normal(0,1,(2,self.nrSteps,self.nrAreas))


    def start_sim(self):
        
        # setup connectivity matrix
        self.set_connectivity()

        # let the integration happen!
        self.output_y, self.output_noise = self.integrator_RK4()
        self.output_y = np.moveaxis(self.output_y, 0, -2)
        self.save_output()

    def set_connectivity(self):
        '''
        In this function the connectivity matrix is setup 
        '''
        self.c_matrix = pd.read_csv('MODEL_Cmatrix_grouped_cre-True_hemi-3_grouping-max.csv')
        self.c_matrix.drop('Unnamed: 0', inplace=True, axis=1)
        self.c_matrix = np.array(self.c_matrix)
        np.fill_diagonal(self.c_matrix, 0)
        print(self.c_matrix)

    def derivatives(self, y, n, par):
        '''
        Parameter:
        y : array, 3 x nrAreas time series of the firing rate values of E, I and A. 
        n : array, 3 x nrAreas time series of noise (Ornstein-Uhlenbeck).
        par : parameter object
        returns : dy, 3 x nrAreas element array with the derivatives of E, I and A
        '''

        dy = np.zeros((3,self.nrAreas))

        # derivative of E - rate
        # aux is a dummy variable for part of the derivative
        #print('y', y[0].shape)
        aux = par.Jee*y[0] - par.Jei*y[1] + par.thetaE -y[2] + n[0] + self.par.G * np.matmul(self.c_matrix, y[0])  

        for area in np.arange(self.nrAreas):
            if aux[area] <= par.Edesp:
                dy[0][area] = -y[0][area]/par.tauE
            else:
                dy[0][area] = (-y[0][area] + par.Eslope*(aux[area]-par.Edesp))/par.tauE
            
        # derivative of I - rate 
        aux = par.Jie*y[0] - par.Jii*y[1] + par.thetaI + n[1]
        for area in np.arange(self.nrAreas):
            if aux[area] <= par.Idesp:
                dy[1][area] = -y[1][area]/par.tauI
            else:
                dy[1][area] = (-y[1][area]+par.Islope*(aux[area]-par.Idesp))/par.tauI

        # derivative of A - adaptation rate
        dy[2] = (-y[2]+par.betaE*y[0])/par.tauAdapt

        #print('dy', dy)
        return dy


    def integrator_RK4(self):
        '''
        The firing rate is computed analytically by using the Runge-Kutta method.  
        '''

        dt = self.sim_params[2]
        t_end = self.sim_params[1]

        noiseDummy1 = np.exp(-dt/self.par.tauN)
        noiseDummy2 = math.sqrt(((2*(self.par.sigmaN**2)/self.par.tauN)*self.par.tauN*0.5)*(1-math.exp(-dt/self.par.tauN)**2))
        rk4Aux1=dt*0.500000000 #(1/2)
        rk4Aux2=dt*0.166666666 # (1/6)
        Tdt = int(1/dt)
        tsteps = np.arange(dt, t_end, dt)

        y_current = self.initial_cond # keeps track of the current value for the rate 
        noise_current = self.noise_init
        noise = [] 
        noise.append(noise_current)
        # time series per population/variable
        y = []
        y.append(y_current)

        # time counter for when to save the computed value
        k = 0 

        # for every time step, we now have to find the solution of the derivative by integrating 
        for iter, step in enumerate(tsteps):

            # calculate the derivatives
            aux1 = self.derivatives(y_current, noise_current, self.par)
            aux2 = y_current + rk4Aux1 * aux1
    
            aux3 = self.derivatives(aux2, noise_current, self.par)
            aux2 = y_current + rk4Aux1 * aux3

            aux4 = self.derivatives(aux2, noise_current, self.par)
            aux2 = y_current + dt * aux4

            aux4 += aux3

            aux3 = self.derivatives(aux2, noise_current, self.par)

            y_current = y_current + rk4Aux2 * (aux1+aux3 + 2*aux4)

            # noise for every area individually
            #print('noise', noise_current.shape)
            noise_current[0] = noise_current[0] *noiseDummy1+noiseDummy2*self.random_vals[0,iter,:]
            noise_current[1] = noise_current[1] *noiseDummy1+noiseDummy2*self.random_vals[1,iter,:]
            
            
            k+= 1
            if k == Tdt:
                y.append(y_current)
                noise.append(noise_current)
                
                k = 0 
                
        return np.array(y), np.array(noise).T

# TODO: write this for comparison!
    def integrator_euler(self):
        '''
        The firing rate is computed analytically by using the Euler method. This
        is a bit less precise than the RK4 method.
        '''

    def plot_results(self):
        print('size output', self.output_y.shape, self.output_noise.shape)
        plt.plot(self.output_y[0], label='E')
        plt.plot(self.output_y[1], label='I')
        #plt.plot(self.output_y[2], label='A')
        plt.legend()
        plt.show()
        # print(self.output_noise[0])

        plt.plot(self.output_y[2], label='A')
        plt.legend()
        plt.show()

        #plt.plot(self.output_noise[0], label='noise 1')
        #plt.plot(self.output_noise[1], label='noise 2')
        #plt.show()

    def save_output(self):
        
        file_addon = f'_14areas_G{self.par.G}_cmatrix'
        print(self.output_y.shape)
        f_rate_E = pd.DataFrame(self.output_y[0])
        f_rate_I = pd.DataFrame(self.output_y[1])
        f_rate_A = pd.DataFrame(self.output_y[2])
        f_rate_E.to_csv(self.output_dir+f'frateE{file_addon}.csv', index=False)
        f_rate_I.to_csv(self.output_dir+f'frateI{file_addon}.csv', index=False)
        f_rate_A.to_csv(self.output_dir+f'frateA{file_addon}.csv', index=False)
        
        #noise_df = pd.DataFrame(self.output_noise)
        #y_df.to_csv(self.output_dir)
        #noise_df



