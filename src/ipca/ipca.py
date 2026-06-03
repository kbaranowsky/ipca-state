"""
Instrumented Principal Component Analysis (IPCA) Estimator.

Estimates the IPCA model of Kelly et al. (2019). The instruments are used as proxy for (latent) exposure risk. 
Following Kelly et al. (2019) the model can include: latent factors, observable factors and pricing errors.
Additionally, the estimator includes the bootstrap test for significance of characteristics related pricing errors, 
charactersitics significance under the Wild Bootstrap procedure, and additionally implements the Dependent Wild Bootstrap procedure
for the above-mentioned test, to account for possible autocorrelations of residuals following thesis implementation.

Finally, the model incluedes the out of sample forecasting, and the performance comparison with Diebold Mariano test, based on the 
squared forecast errors as the loss differential function.


Main class
----------
IPCA
    Estimator for restricted and unrestricted IPCA models.

References
----------
Baranowski, K. (2026). State Conditioning in Instrumented Principal
Component Analysis: Do Market States Change Characteristics Based Risk
Exposures? Unpublished Bachelor thesis, Erasmus School of Economics, Erasmus University Rotterdam.

Kelly, B. T., Pruitt, S., and Su, Y. (2019). Characteristics are covariances: A unified model of
risk and return. Journal of Financial Economics, 134(3):501–524.    
    
"""




import time
import pandas as pd
import numpy as np
import scipy.linalg as la # optimization
import scipy.sparse.linalg as ssla # optimization
from joblib import Parallel, delayed # parallelization
import statsmodels.api as sm
from scipy.stats import norm



class ipca(object):

    """

    Instrumented Principal Component Analysis (IPCA) Estimator.

    Estimates the IPCA model of Kelly et al. (2019). The instruments are used as proxy for (latent) exposure risk. 
    Following Kelly et al. (2019) the model can include: latent factors, observable factors and pricing errors.
    Additionally, the estimator includes the bootstrap test for significance of characteristics related pricing errors, 
    charactersitics significance under the Wild Bootstrap procedure, and additionally implements the Dependent Wild Bootstrap procedure
    for the above-mentioned test, to account for possible autocorrelations of residuals following thesis implementation.


    Parameters
    ----------
    Z : dict
        Dictionary with asset characteristics. Each key is a time period and each value is a ``N_t * L`` pandas DataFrame, where
        rows are assets and columns are firms characteristics.
    R : dict, optional
        Dictionary with asset returns. Each key is a time period and each value is a pandas Series of length ``N_t``. It is required, 
        unless ``X`` (managed portfolios returns) are supplied.
    X : dict, optional
        Dictionary with maanged portfolio returns. Each key is a time period and each value is a pandas Series of length ``L``. If
        it is not supplied, managed portfolio returns are constrcted from ``Z`` and ``R``.
    K : int, default = 0
        Number of latent factors to be included in the model.
    G : pandas DataFrame, optional
        Observed factor returns with factors as rows and time periods as columns.
    alpha: bool, default = False
        If true, it estimates the unrestricted model of Kelly et al. (2019) with an intercept pricing error, which
        is instrumented by the asset characteristics.

        
    
    Attributes
    ---------
    Gamma : pandas DataFrame
        Estimated characteristics to factor loadings matrix.
    F : pandas DataFrame
        Estimated latent factor returns.
    Lambda: pandas Series
        Time series average factor returns used for the predictive fitteed returns in the predictive ``R^2``.
    total_R_squared : float
        In sample total ``R^2``.
    predictive_R_squared : float
        In sample predictive ``R^2``.
    estimated: bool
        Whether the model has been fitted.

    Notes
    ----
    For NOW none. Add later the short description about the ALS estimation. I think it is quite useful to add here.

    """


    def __init__ (self, Z = None, R = None, X = None, K = 0, S = None, G = None, alpha = False):

        """

        Initializes the IPCA estimator based on Kelly et al. (2019)

        The constructuor, checks for the required inputs, and makese the model representation
        before it is estimated. Alignsm the observed factor returns iwht the estimation periods, 
        constructs factor names and prepares the managed portfolio returns for the esitmation.

        Raises
        -------
        ValueError
            If characteristics (Z) are not supplied.
        ValueError
            IF neither asset returns (R) or managed portfolio returns (X) are not supplied.
        ValueError
            If the model contains no latent factors, no observed facotrs and no alpha.

        """

        ###
        # Add here to cut the data into estimation and evalutation parts, so i need to change the constructor, but I want to do that with as little changes as possible 

        # Check the minimal inputs
        if Z is None:
            raise ValueError("Asset characteristics must be provided!")
        
        if R is None and X is None:
            raise ValueError("Asset returns or managed-portfolio returns must be provided!")
        
        if K == 0 and G is None and alpha == False:
            raise ValueError("Model must specify latent factors, observed factors or include constant!")
        

        # Definition of the model
        self.isUnrestricted = False if not alpha else True
        self.hasLatent = True if K > 0 else False
        self.hasObserved = False if G is None else True
        self.stateDependent = False if S is None else True


        # Defining the inputs for objct
        self.Z = Z
        self.R = R
        self.X = X
        self.S = S
        self.G = G
        self.K = K
        self.alpha = alpha

        # Defining the length of the model and characretistics
        self.times = sorted(Z.keys())
        self.T = len(self.times)
        self.characteristics = list(Z[self.times[0]].columns) # Namings of the characteristics
        self.L = len(self.characteristics)


        # Defining names of (latent) factors
        self.latentFactorsNames = [f"F{k}" for k in range(1, self.K + 1)]

        if G is None:
            self.G = None
            self.observedFactorNames = []

        else:
            self.G = G.copy()

            self.G = self.G.loc[:, self.times]
            self.observedFactorNames = list(self.G.index)


        

        if self.alpha:

            # If the model is unrestricted it assumes that the constant is just a given factor
            alpha_row = pd.DataFrame(np.ones((1, self.T)), index = ["alpha"], columns = self.times)

            if self.G is None:
                self.G = alpha_row
                self.observedFactorNames = ["alpha"]
            else:
                self.G = pd.concat([alpha_row, self.G], axis = 0)
                self.observedFactorNames = ["alpha"] + self.observedFactorNames

        # Updaing whether alpha is included if no other observed factors are included in the model
        self.hasObserved = True if not self.G is None else False

        # All factor names
        self.factorNames = self.latentFactorsNames + self.observedFactorNames

        # Defining the number of observed and total factors
        self.O = len(self.observedFactorNames)
        self.NumAllFactors = self.K + self.O


        
        # Making some preprerations for inputs of the ALS
        self.X, self.sigmaZ, self.N_t = self._prepare_inputs()


        # I need to add here some sample size summaries for the long summary func


        # Defining the outputs
        self.Gamma = None
        self.F = None
        self.Lambda = None
        self.pricingErrorStat = None
        self.pricingErrorPValue = None

        self.total_R_squared = None
        self.predictive_R_squared = None




        # Defining summary outputs
        self.estimated = False
        self.converged = None
        self.n_iter = None
        self.final_tol = None
        self.max_iter = None
        self.parallel = None
        self.runtime = None






    def _prepare_inputs(self):

        """
        Prepare the objects that are needed for Alternating Least Squares (ALS).
        
        Returns:
        --------
        X : dictionary
            With T keys and L pandas series with managed portfolios returns.
        sigmaZ : dictionary
            With T keys and L*L pandas dataframes of second moments of asset characteritiscs.
        N_valid : dictionary
            With T keys and L int values with number of valid assets for each estimation period.
        """

        X, sigmaZ, N_valid = {}, {}, {}

        for t in self.times:
            Z_t = self.Z[t]

            if self.X is None:
                R_t = self.R[t]
                N_valid[t] = len(R_t)
                X[t] = Z_t.T.dot(R_t) / N_valid[t]
            else:
                N_valid[t] = len(Z_t)
                X[t] = self.X[t] #/ N_valid[t]

            # Computing the secodn momemtn matrix
            sigmaZ[t] = Z_t.T.dot(Z_t)/ N_valid[t]



        return X, sigmaZ, N_valid
    


    def fit(self, tol = 1e-6, max_iter = 1000, isRestricted = True, printTime = True, printInformation = True, useParraell = False, Gamma0 = None):

        """
        Estimate the IPCA model uisng Alternating Least Squares.

        Initializes the Gamma0 accroding to the requirements of Kelly et al. (2019). And then, 
        iteretively updates latent factotrs and loadings (with the OLS equations) and then
        normalizes them, and finally computes the in-sample fit measures.
        
        Parameters
        ----------
        tol : float, defualt = 1e-6
            Convergence tolerance for the ALS.
        max_iter : int, default = 1000
            Maximum number of ALS iterations.
        isRestricted : bool, default = True
            Leftover, when tried to do anouther implementation. (I need to clean it up)
        printTime : bool, default = True
            Whether to print the model runtime afreer estimation.
        printInformation:
            Whether to print ALS iteration information.
        useParraell : bool, default = False
            Leftoover, when tried to implement parallelization here (but I did it in bootstraps insread, and its much smarter, so 
            also need to clean this one up.).
        Gamma0 : pandas dataframe, optional
            Optional starting value for the Gamma matrix.
        
        Returns
        -------
        ipca
            Estimated model
        """

        # Start the timer
        start_time = time.time()


        # Initializing the parameters 
        Gamma0_SVD, F0 = self._initialize_param()

        # makeing possible starting gamma other than the kelly implementation, ( I use it later for the forecasts, and that is why I added it)
        if Gamma0 is None:
            Gamma0 = Gamma0_SVD
        else:
            Gamma0 = pd.DataFrame(Gamma0.values, index = self.characteristics, columns = self.factorNames)



        if self.hasLatent:
            # Actual model for iteration
            Gamma, F, n_iter, converged, final_tol = self._run_als(Gamma0 = Gamma0, tol = tol, max_iter = max_iter, printInformation = printInformation)
        else: 
            F = F0
            Gamma = self._update_gamma(F)
            n_iter = 1
            converged = True
            final_tol = 0.0
        
        # Update the instance variable sof the object
        self.Gamma = Gamma
        self.F = F

        # Getting the time series average of the factor returns to calucalte the R sqauared results

        lambdaParts = []
        
        if self.hasLatent:
            lambdaParts.append(self.F.mean(axis = 1))

        if self.hasObserved:
            lambdaParts.append(self.G.mean(axis = 1))

        self.Lambda = pd.concat(lambdaParts).loc[self.factorNames]

        # getting the model fits
        if self.R is not None:
            self.total_R_squared = self._get_total_R_squared()
            self.predictive_R_squared = self._get_predictive_R_squared()
            self.total_R_squared_manged = self._get_total_R_squared_managed()
            self.predictive_R_squared_managed = self._get_predictive_R_squared_managed()

        else:
            self.total_R_squared = self._get_total_R_squared_managed()
            self.predictive_R_squared = self._get_predictive_R_squared_managed()


        # Estimation diagnostic update
        self.n_iter = n_iter
        self.converged = converged
        self.final_tol = final_tol
        self.max_iter = max_iter
        self.parallel = useParraell
        self.runtime = time.time() - start_time
        self.estimated = True

        # Pringing the time it took to run the model
        if printTime:
            print(f"It took {self.runtime:.2f} seconds for the model to run.")

        return self


    def long_summary(self):

        """
        Prints a long summary of the IPCA model, inclduing the diagnostics data.
        
        """




        print("=" * 60)
        print("IPCA model with {self.K} factors.")
        print("=" * 60)

        print("Summary of fit:")
        print("=" * 60)
        print(f"Total R2 (individual assets): {self.total_R_squared}")
        print(f"Predictive R2 (individual assets): {self.predictive_R_squared}")
        print(f"Total R2 (managed portfolios): {self.total_R_squared_manged}")
        print(f"Predictive R2 (managed portfolios): {self.predictive_R_squared}")
        print("=" * 60)
        print("Detailed diagnostics:")
        print("=" * 60)
        print(f"Estimated: {self.estimated}")
        print(f"Converged: {self.converged}")
        print(f"Runtime: {self.runtime}")
        print(f"Maximum number of iterations: {self.max_iter}")
        print(f"Number of iterations: {self.n_iter}")
        print(f"Final tolerance: {self.final_tol}")
        print(f"Used parallel estimaiton: {self.parallel}")
        

        

    def short_summary(self):
        """

        Prints a short summary of the IPCA model.

        """

        print("=" * 60)
        print("IPCA model with {self.K} factors.")
        print("=" * 60)

        print("Summary of fit:")
        print("=" * 60)
        print(f"Total R2 (individual assets): {self.total_R_squared}")
        print(f"Predictive R2 (individual assets): {self.predictive_R_squared}")
        print(f"Total R2 (managed portfolios): {self.total_R_squared_manged}")
        print(f"Predictive R2 (managed portfolios): {self.predictive_R_squared}")







    def _run_als(self, Gamma0 = None, tol = 1e-6, max_iter = 1000, printInformation = True, useParraell = False):

        """
        
        Runs the main ALS loop.
  
        
        """

        # Initialization of the elements
        Gamma = Gamma0.copy()
        F = None

        # Things for the loop
        iteration = 1
        tolerance = np.inf
        converged = False

        while iteration <= max_iter and tolerance > tol:
           
           iteration += 1
           
           Gamma_old = Gamma.copy()

           Gamma, F = self._single_als_iteration(Gamma_old)

           tolerance = np.linalg.norm(Gamma.values - Gamma_old.values) / max(1.0, np.linalg.norm(Gamma_old.values))

           if tolerance <= tol:
               converged = True

           if printInformation and (iteration == 1 or iteration % 10 == 0 or tolerance <= tol):
               print(f"Iteration {iteration}: tolerance = {tolerance: .6e}")
        
        n_iter = iteration 
        final_tol = tolerance

        return Gamma, F, n_iter, converged, final_tol


    
    def _initialize_param(self):

        """
        Initialoizes the Gamma and F in the first step of thje ALS estimation.

        """




        # Creates the inital Gamma 0 of zeros, but it a a basically combination of gamma beta and gamma delta
        # It is better to do this, becuase unrestricted model needs additional gamma, instead of doing 2 or 3 matricies seperately, it is kind of a joint one
        Gamma0 = pd.DataFrame( 0.0, index = self.characteristics, columns = self.factorNames)

        if self.hasLatent:

            # Geeting the X matrix from the X dictionary
            X_matrix = pd.DataFrame({t: self.X[t] for t in self.times}).values

            U, Gam, Vt = la.svd(X_matrix, full_matrices = False)

            Gamma0.loc[:, self.latentFactorsNames] = U[:, :self.K]

            F0 = pd.DataFrame( np.diag(Gam[:self.K]).dot(Vt[:self.K, :]), index = self.latentFactorsNames, columns = self.times)
        else:

            F0 = pd.DataFrame( index = self.latentFactorsNames, columns = self.times)


        return Gamma0, F0
    








    def _single_als_iteration(self, Gamma0 = None):

        """
        Makse a single als loop: estimate factors, estimate gamma
        """
        
        
        
        # Compute next iteration
        F = self._update_factors(Gamma0)
        Gamma = self._update_gamma(F)

        # Normalize the results
        if self.hasLatent:
            Gamma_latent, F = self._normalize(Gamma.loc[:, self.latentFactorsNames], F)

            # Update the latent factos gamma
            Gamma.loc[:, self.latentFactorsNames] = Gamma_latent

            # I forgot to normalize alpha initially
            if self.alpha:
                Gamma, F = self._normalize_gamma_alpha(Gamma, F)

            # Correcting the signs and update again
            Gamma_latent, F = self._fix_sign(Gamma.loc[:, self.latentFactorsNames], F)
            Gamma.loc[:, self.latentFactorsNames] = Gamma_latent

        

        return Gamma, F


    









    def _update_factors(self, Gamma):

        """
        Updates the factor returns based on given Gamma mapping.

        
        """


        # Initialie the factors dataframe
        factors = pd.DataFrame(index = self.latentFactorsNames, columns = self.times)

        # If the model is only observed factors we do not estimate them 
        if not self.hasLatent:
            return factors # Returns empty dataframe
        

        # Splitting the gamma matrix 
        Gamma_beta = Gamma.loc[:, self.latentFactorsNames].values

        if self.hasObserved:
            Gamma_delta = Gamma.loc[:, self.observedFactorNames].values


        for t in self.times:
            X_t = self.X[t].values
            sigmaZ_t = self.sigmaZ[t].values

            if self.hasObserved:
                G_t = self.G.loc[self.observedFactorNames, t].values
                X_t = X_t - sigmaZ_t.dot(Gamma_delta).dot(G_t)
            
            numerator = Gamma_beta.T.dot(X_t)
            denominator = Gamma_beta.T.dot(sigmaZ_t).dot(Gamma_beta)

            # Get the final factors
            factors[t] = self._get_factors_without_inverse(numerator = numerator, denominator = denominator)

        
        
        return factors 
    




    def _update_gamma(self, F):

        """
        Updates the gamma dataframe based on given set of factors.
        
        """


 
        # Full gamma matrix is passed 
        vector_representation_gamma_len = self.L * self.NumAllFactors


        # Initializing the objects for calucalting the kronecker product equation
        left_inverse = np.zeros((vector_representation_gamma_len, vector_representation_gamma_len))
        right_part = np.zeros(vector_representation_gamma_len)

        # Initializing the vector representation of the new estimated gamma
        vec_rep_gamma = np.zeros(vector_representation_gamma_len)

        
        for t in self.times:

            factorParts = []

            if self.hasLatent:
                factorParts.append(F.loc[self.latentFactorsNames,t])

            if self.hasObserved:
                factorParts.append(self.G.loc[self.observedFactorNames,t])

            allFactorsReturns = pd.concat(factorParts)


            left_inverse += np.kron(self.sigmaZ[t].values, np.outer(allFactorsReturns.values, allFactorsReturns.values)) * self.N_t[t]

            right_part += np.kron(self.X[t].values, allFactorsReturns.values) * self.N_t[t]

        





        vec_rep_gamma = self._get_gamma_without_inverse(numerator = right_part, denominator = left_inverse)


        # Now reshaping to the previous gamma size

        Gamma_final = pd.DataFrame( vec_rep_gamma.reshape(self.L, self.NumAllFactors), index = self.characteristics, columns = self.factorNames)

        return Gamma_final



    
    def _normalize(self, Gamma, F):

        """
        Normalize the gamma and F according to the restrictions in the paper. It is esentially an equivalence of what we normally done for the PCA.

        Input:
        Gamma: is an L*K characteristics to factor loadings mapping in the form of panda data frame.
        F: is an K*T panda data frame of factors for each time period in the model.
        
        Output:
        Gamma_norm: normalized gamma
        F_norm: normalized factors
        
        
        """


        # Extracting the arrays
        G = Gamma.values.astype(float)
        F_val = F.values.astype(float)

        ## Restriction 1 (orthonormaliza gamma)
        L = la.cholesky(G.T.dot(G), lower = False)
        G_orthonormal = G.dot(la.inv(L))

        # Rotate factors accrodingly
        F_rotated = L.dot(F_val)

        # Make sigma diagonal and decreasing in values
        Sigma = F_rotated.dot(F_rotated.T)

        #Spectral decomposition on the sigma matrix
        U, Gam, V = la.svd(Sigma)

        G_norm = G_orthonormal.dot(U)
        F_norm = U.T.dot(F_rotated)


        # Flipping signs
        Gamma_norm = pd.DataFrame( G_norm, index = Gamma.index, columns = Gamma.columns)

        F_norm = pd.DataFrame( F_norm,  index = F.index,  columns = F.columns)
        
    

        return Gamma_norm, F_norm
    





    @staticmethod
    def _fix_sign(Gamma, F):
        """
        Changes the signs of gamma and F such that the factors returns are always positive.
        
        Input:
        Gamma: L*K panda dataframe with characteristics to factor loading map.
        F: K*T panda dataframe with K (latent) factors for T time periods.
        
        Output:
        Gamma_final: corrected mapping.
        F_final: corrected factors.
        
        
        """

    

        # copying for security
        Gamma_final  = Gamma.copy()
        F_final = F.copy()

        factorMeans = F_final.mean(axis = 1)

        # I am doing it for now with for loop which may be hella inefficienct, but tbh usually K is small so it should not make that big of a difference?
        # Test it out if I have very slow code, there should be an R equivalence of lapply() or something simmilar
        # Swapping the signs
        for k in range(len(factorMeans)):
            if factorMeans.iloc[k] < 0:
                Gamma_final.iloc[:, k] = -Gamma_final.iloc[:, k]
                F_final.iloc[k, :] = -F_final.iloc[k, :]



        return Gamma_final, F_final
    
    

    def _normalize_gamma_alpha(self, Gamma, F):

        """
        
        Normalies alpha matrix, with respect to gamma beta matrix, when the pricing error is present in the model.
        
        """

        if not self.alpha:
            return Gamma, F
        
        if not self.hasLatent:
            return Gamma, F
        
        if "alpha" not in Gamma.columns:
            return Gamma, F
        
        Gamma_out = Gamma.copy()
        F_out = F.copy()


        Gamma_beta = Gamma_out.loc[:, self.latentFactorsNames].values.astype(float)
        gamma_alpha = Gamma_out.loc[:, "alpha"].values.astype(float)

        c = Gamma_beta.T.dot(gamma_alpha)

        gamma_alpha_ortho = gamma_alpha - Gamma_beta.dot(c)

        F_out.loc[self.latentFactorsNames, :] = F_out.loc[self.latentFactorsNames, :].values + c.reshape(-1,1)

        Gamma_out.loc[:, "alpha"] = gamma_alpha_ortho



        return Gamma_out, F_out




    @staticmethod
    def _get_factors_without_inverse(numerator, denominator):

        """
        This function calculates the factors without using the inverse of the denominator matrix. It is quicker and more stable than just calculating the precision matirx.
        
        Formula: Ax = B, where A is the denominator matrix, x is the factors that we want, and B is the numerator matrix.
        Inputs:
        numerator: is a K*1 matrix that represents the numerator of the equation for calculating the factors.
        denominator: is a K*K matrix that represents the denominator of the equation for calculating the factors.

        Output:
        factors: is a K*1 vector that represents the calcualted factors.
        
        """

        factors = np.linalg.lstsq(denominator, numerator, rcond = None)[0]

        return factors 
    



    
    @staticmethod
    def _get_gamma_without_inverse(numerator, denominator):

        '''
        This function calcualtes the gamma matrix using the inverse of the denominator matrix. It is equivalence to the method for factors, but here it is prepared for eq. (7) from paper.
        
        Inputs: 
        numerator: is a KL*1 matrix that represents the numerator of the equation for calculating the vector representation of gamma.
        denominator: is a LK*LK matrix that represents the denominator...
        '''


        gamma = np.linalg.lstsq(denominator, numerator, rcond = None)[0]

        return gamma





    # Calculating the performance of the model

    def _get_total_R_squared(self):

        """
        Function that caluclates the IPCA total R^2 based on equation 15 from the paper.
        """

       

        # Defining the numerator and denominator
        ssr, sst = 0.0, 0.0
        Gamma = self.Gamma.values


        for t in self.times:

            Z_t = self.Z[t]
            R_t = self.R[t]

            factorParts = []

            if self.hasLatent:
                factorParts.append(self.F.loc[self.latentFactorsNames, t])

            if self.hasObserved:
                factorParts.append(self.G.loc[self.observedFactorNames, t])

            allFactorsReturns = pd.concat(factorParts).values

            fitted = Z_t.values.dot(Gamma).dot(allFactorsReturns)
            residual = R_t.values - fitted

            ssr += residual.T.dot(residual)
            sst += R_t.values.T.dot(R_t.values)


        return 1 - ssr / sst
    




    def _get_predictive_R_squared(self):

        """
        Function that caluclates the IPCA predictive R^2 based on equation 16 from the paper.
        
        """


        # Defining the numerator and denominator
        ssr, sst = 0.0, 0.0
        Gamma = self.Gamma.values
        Lambda = self.Lambda.loc[self.factorNames].values


        for t in self.times:

            Z_t = self.Z[t]
            R_t = self.R[t]

            fitted = Z_t.values.dot(Gamma).dot(Lambda)
            residual = R_t.values - fitted

            ssr += residual.T.dot(residual)
            sst += R_t.values.T.dot(R_t.values)


        return 1 - ssr / sst
    






    def _get_total_R_squared_managed(self):

        """
        Function that calucaltes the managed portfolio version of totoal R2.
        
        """



        ssr, sst = 0.0, 0.0
        Gamma = self.Gamma.loc[:, self.factorNames].to_numpy(float)

        for t in self.times:
            X_t = self.X[t].loc[self.characteristics].to_numpy(float)
            W_t = self.sigmaZ[t].loc[self.characteristics, self.characteristics].to_numpy(float)

            parts = []



            if self.hasLatent:
                parts.append(self.F.loc[self.latentFactorsNames, t])
            if self.hasObserved:
                parts.append(self.G.loc[self.observedFactorNames, t])

            f_t = pd.concat(parts).loc[self.factorNames].to_numpy(float)

            resid = X_t - W_t.dot(Gamma).dot(f_t)

            ssr += resid.dot(resid)
            sst += X_t.dot(X_t)

        return 1.0 - ssr / sst


    def _get_predictive_R_squared_managed(self):

        """
        Function that calcualtes the manged portoflio version of the predictive R2.
        """


        ssr, sst = 0.0, 0.0
        Gamma = self.Gamma.loc[:, self.factorNames].to_numpy(float)
        Lambda = self.Lambda.loc[self.factorNames].to_numpy(float)

        for t in self.times:
            X_t = self.X[t].loc[self.characteristics].to_numpy(float)
            W_t = self.sigmaZ[t].loc[self.characteristics, self.characteristics].to_numpy(float)

            resid = X_t - W_t.dot(Gamma).dot(Lambda)

            ssr += resid.dot(resid)
            sst += X_t.dot(X_t)

        return 1.0 - ssr / sst















    def test_alpha_pricing_error(self, B = 100, tol = 1e-6, max_iter = 500, mode = "WB", seed = None, restricted_model = None, useParallel = False, n_jobs = -1, printUpdates = True):


        """
        This is the new test alpha function, where I incorporate the parallelization and mode with dependent wild bootstrap.


        Parameters
        -----------
        B : int, default = 100
            Number of bootstrap replciations.
        tol : float, defautl = 1e-6
            Convergence tolerance for ALS in the bootstrap models.
        max_iter : int, defualt = 500
            Maximum number of ALS iterations for bootstraps.
        mode : ["WB", "DWB"], defualt = "WB"
            Model of the Bootstrap, either Wild Bootstrap or Dependent Wild Bootstrap.
        seed : int, optional 
            Seed inclided for reporducibilityu.
        restrictedModel : ipca, optional
            Preestimated restricted model under the null. For quicker running, so I do not have to re run it again.
        useParallel : bool, default = False
            Whehter to run the bootstraps under parallelizations scheme.
        n_jobs : int, default = - 1
            Number of parallel jobs - how many cors can be used. Default is all.
        printUpdates : bool, default = True
            Whether to print updates of the bootstap progress and the running p vlaues.

        Returns
        -------
        p_value : float
            Bootstrap p-valeue.
        
        Raises
        -------
        ValueError
            If the model has not been estimated.
        ValueError 
            If the model is restricted.
        ValueError
            If the modl has no latent facotrs,
        ValueError
            If the observed facotrs other than alpha are included.
        ValueError
            If mode is not Wild Bootstrap or Depednet Wild Bootstrap.
        """


        # Adding start time to show long long the bootstrap method was working
        start_time_whole_test = time.time()




        # Adding basic checks to throw - locks when the method can be actually called
        if not self.estimated:
            raise ValueError("The test can only be run on estimated unrestricted model.")
        if not self.alpha:
            raise ValueError("This test can only be run on unrestricted model where alpha is present")
        if not self.hasLatent:
            raise ValueError("The implementation is for the latent factor only.")
        if any(name != "alpha" for name in self.observedFactorNames):
            raise ValueError("This implementation does not support the observed factors ")
        if mode not in ["WB", "DWB"]:
            raise ValueError("Only WB and DWB modes are available")
        

        # Step 1) get the statistic for the unrestricted model - common for all modes and parralel or not
        gamma_alpha_vec = self.Gamma.loc[:, "alpha"].to_numpy(dtype = float)
        W_alpha = float(gamma_alpha_vec.T.dot(gamma_alpha_vec))
        self.pricingErrorStat = W_alpha

        # First step of creating the managed portfolios - estimate restricted model if it was not given
        if restricted_model is None:
            restricted_model = self.__class__(Z = self.Z, R = None, X = self.X, K = self.K, G = None, alpha = False)
            restricted_model.fit(tol = tol, max_iter = max_iter, printTime = False, printInformation = False, useParraell = False)
        else:
            if not restricted_model.estimated:
                restricted_model.fit(tol = tol, max_iter = max_iter, printTime = False, printInformation = False, useParraell = False)

        # stack the managed portfolios to the matrix T by L
        X_matrix = np.vstack([self.X[t].loc[self.characteristics].to_numpy(dtype = float) for t in self.times])

        restricted_model_fit = self._managed_fit_matrix_alpha(restricted_model)
        unrestricted_model_fit = self._managed_fit_matrix_alpha(self)

        # Get the residuals from the unrestricted model
        RESID = X_matrix - unrestricted_model_fit

        # Setting rng for replication
        rng = np.random.default_rng(seed)


        # darwing random indicies adn q's for WB or the Depednet Wild bootstrap
        boot_indices = np.empty((B, self.T), dtype = int)
        boot_q = np.empty((B, self.T), dtype = float)

        if mode == "WB":


            
            
            for b in range(B):
                boot_indices[b, :] = rng.integers(low = 0, high = self.T, size = self.T)
                boot_q[b, :] = rng.standard_t(df = 5, size = self.T) / np.sqrt(5.0 / 3.0)

        elif mode == "DWB":

            boot_indices = None
            ell = 6

            index = np.arange(self.T)
            distance = np.abs(index[:, None] - index[None, :])
            Sigma = np.maximum(1.0 - distance / ell, 0.0)


            for b in range(B):
                boot_q[b, :] = self._draw_bartlett_multiplier(T = self.T, rng = rng, ell = ell, Sigma = Sigma)












        # Save main results
        W_boot = np.empty(B, dtype=float)


        if useParallel:

            #print("Not yet implemented")
            if printUpdates:
                print(f"Running alpha bootstrap in parallel with B = {B}, n_jobs = {n_jobs} and mode = {mode}")

            results = Parallel(
                n_jobs = n_jobs, 
                backend = "loky"
            )(
                delayed(self._single_alpha_bootstrap)(b = b, restricted_model = restricted_model, restricted_model_fit = restricted_model_fit, RESID = RESID, mode = mode, sampled_indices = None if boot_indices is None else boot_indices[b, :], q = boot_q[b, :], max_iter = max_iter, tol = tol) for b in range(B)
            )

            W_boot[:] = np.array(results, dtype=float)



        else:


            for b in range(B):
                W_boot[b] = self._single_alpha_bootstrap(b = b, restricted_model = restricted_model, restricted_model_fit = restricted_model_fit, RESID = RESID, mode = mode, sampled_indices = None if boot_indices is None else boot_indices[b, :], q = boot_q[b, :], max_iter = max_iter, tol = tol)

                # Prining updates 
                if printUpdates and ((b + 1) % max(1, B //10) == 0 or b == B - 1):
                    running_p = float(np.mean(W_boot[:b + 1] >= W_alpha))

                    print(f"Bootstrap {b+1:>5}/{B}")
                    print(f"Running p-value = {running_p:.6f}")

            

        # Compute final p value
        p_value = float(np.mean(W_boot >= W_alpha))
        self.pricingErrorPValue = p_value



        return p_value
    




    

    @staticmethod
    def _resample_single_boot_alpha_WB(restricted_model_fit, RESID, sampled_indices, q):
        """
        Resmaples single data under the null in the bootstrap test for pricing error (with Wild Bootstrap).
        """


        X_boot_matrix = restricted_model_fit + RESID[sampled_indices, :] * q[:, None]


        return X_boot_matrix
    
    @staticmethod
    def _resample_single_boot_alpha_DWB(restricted_model_fit, RESID, q):

        """
        Resmaples single data under the null in the bootstrap test for pricing error (with Dependent Wild Bootstrap).
        """

        X_boot_matrix = restricted_model_fit + RESID * q[:, None]

        return X_boot_matrix
    





    def _single_alpha_bootstrap(self, b, restricted_model, restricted_model_fit, RESID, mode, sampled_indices, q, max_iter, tol):

        """
        Runs a single bootstrap for pricing error test.
        """

       

        if mode == "WB":
            X_boot_matrix = self._resample_single_boot_alpha_WB(restricted_model_fit = restricted_model_fit, RESID = RESID, sampled_indices = sampled_indices, q = q)
        elif mode == "DWB":
            X_boot_matrix = self._resample_single_boot_alpha_DWB(restricted_model_fit = restricted_model_fit, RESID = RESID, q = q)
        else:
            raise RuntimeError("This shouldnt have happened. As only two modes are allowed")
        
        
        # Convert it to a dictionary
        X_boot = {t: pd.Series(X_boot_matrix[i, :], index = self.characteristics, name = t) for i, t in enumerate(self.times)}

        #Make the bootstrap model
        boot_model = self.__class__(Z = self.Z, R = None, X = X_boot, K = self.K, G = None, alpha = True)

        # match starting values of the authros
        Gamma0 = pd.DataFrame(0.0, index = self.characteristics, columns = boot_model.factorNames)
        Gamma0.loc[:, boot_model.latentFactorsNames] = (restricted_model.Gamma.loc[self.characteristics, restricted_model.latentFactorsNames].to_numpy(dtype = float))
        F0 = restricted_model.F.loc[restricted_model.latentFactorsNames, self.times].copy()

        # Estiamte the model
        Gamma_b, F_b,n_iter_b, converged_b, final_tol_b = (self._run_unrestricted_from_restricted_start(model = boot_model, Gamma_start= Gamma0, F_start = F0, 
                                                                                                   max_iter = max_iter, tol = tol))
        

        # statistic value form the bootstrap
        gamma_alpha_b = Gamma_b.loc[:, "alpha"].to_numpy(dtype=float)
        W_b = float(gamma_alpha_b.T.dot(gamma_alpha_b))



        return W_b
    
    

    def _run_unrestricted_from_restricted_start(self, model, Gamma_start, F_start, max_iter, tol):


        """
        Runs the ALS initialized in the Gamma start and F start.
        """

        Gamma_old = Gamma_start.copy()
        F_old = F_start.copy()

        # Helper variables for the diagnositscc later
        converged = False
        final_tol = np.inf
        n_iter = 0

        while n_iter <= max_iter and final_tol > tol:
            Gamma_new, F_new = model._single_als_iteration(Gamma_old)

            gamma_change = np.max(np.abs(Gamma_new.to_numpy(dtype = float) - Gamma_old.loc[Gamma_new.index, Gamma_new.columns].to_numpy(dtype = float)))

            f_change = np.max(np.abs(F_new.to_numpy(dtype = float) - F_old.loc[F_new.index, F_new.columns].to_numpy(dtype = float)))

            final_tol = max(gamma_change, f_change)

            Gamma_old = Gamma_new.copy()
            F_old = F_new.copy()

            n_iter += 1
            
            if final_tol <= tol:
                converged = True
                break



        return Gamma_old, F_old, n_iter, converged, final_tol
    
    
    @staticmethod
    def _managed_fit_matrix_alpha(model):

        """
        Returns the T * L matrix of fitted managed portofolio returns for pricing error test.
        """

        Gamma = model.Gamma.loc[model.characteristics, model.factorNames].to_numpy(dtype = float)

        fitted = np.empty((model.T, model.L), dtype = float)

        for i, t in enumerate(model.times):
            W_t = model.sigmaZ[t].loc[model.characteristics, model.characteristics].to_numpy(dtype = float)

            factor_parts = []

            if model.hasLatent:
                factor_parts.append(model.F.loc[model.latentFactorsNames, t])

            if model.hasObserved:
                factor_parts.append(model.G.loc[model.observedFactorNames, t])

            f_t = pd.concat(factor_parts).loc[model.factorNames].to_numpy(dtype = float)

            fitted[i, :] = W_t.dot(Gamma).dot(f_t)


        return fitted
    


    
    @staticmethod
    def _draw_bartlett_multiplier(T, rng, ell = 6, Sigma = None):
        
        """
        Draws the random multipliers form the barlett kernel function. 
        """

        # Creatnig the covariance matrix for the Bartlet kernel function
        if Sigma is None:
            index = np.arange(T)
            distance = np.abs(index[:, None] - index[None, :])

            Sigma = np.maximum(1.0 - distance/ ell, 0.0)

        # it was not numerically stable so I need to make these adjiustments (I cannot use the rng.multivariate normal)
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma= Sigma + 1e-10 + np.eye(T)

        L = np.linalg.cholesky(Sigma)
        z = rng.standard_normal(T)
        q = L.dot(z)


        return q




    # this is the gereneralization of multiple characteritscs test and individual test, tehcinically you can test alpha with it as well
    # actuallly not true lmao


    def test_characteristics(self, B = 100, tol = 1e-6, max_iter = 500, mode = "WB", test_chars = None, model = "baseline",
                              seed = None, useParallel = False, n_jobs = -1, printUpdates = True):
        
        """

        Bootstrap test for individual or joint characteristics releveance.

        Tests whether selected characteristics significantlyy conturbute to the model.


        Parameters
        ----------
        B : int, default = 100
            Numnber of bootstraps.
        tol : float, defautl = 1e-6
            Convergence tolerance for ALS.
        max_iter : int, defatl = 500
            Maximum number of ALS iteratiosn.
        mode : ["WB", "DWB"], defualt = "WB"
            Bootstap model type, either Wild Bootstrap or Dependent Wild Bootstrap.
        test_chars : str of list of str
            Characteristic or list of charactersitics to be tested.
        model : ["baseline", "state"], default = "baseline"
            Added as a control.
        seed : int, optional
            Seed added for reproducibility.
        useParallel : bool, default - False
            Whether to run bootstrap replications in parallel.
        n_jobs : int, default = - 1
            Number of cores used for bootstrap calucaltsions.
        printUpdates : bool, defautl = True
            Whether to print bootstrap progress and runnning p-values.

        Returns
        -------
        tuple
            tuple with p_value on the first position, and the change in total R2 as the second positon.

        Raises
        -------
        ValueError
            If list of characteristics to test is not provided.
        ValueError
            If the model has not been estimated.
        ValueError
            If the model has no latent factors.
        ValueError
            If the model is unrestricted.
        ValueError
            If the mode is not WB or DWB.
        ValueError
            If the model is not baseline or state.
        ValueError
            If any provided charactersitics are not present in the model.

        
        """
        
        # Again basic checks
        
        if test_chars is None:
            raise ValueError("At least one characteristics name is needed to use this test")
        
        if not self.estimated:
            raise ValueError("The model needs to be first estimated and then the test can be called")
        
        if not self.hasLatent:
            raise ValueError("The implementation is for the latent factors only")
        
        if self.alpha:
            raise ValueError("The test is implemented only for restrcited model")
        
        if mode not in ["WB", "DWB"]:
            raise ValueError("Only WB and DWB modes are available")
        
        if model not in ["baseline", "state"]:
            raise ValueError("Choose either baseline or state model specification to get the correct data")
        

        # this change for a single characteristic
        if isinstance(test_chars, str):
            test_chars = [test_chars]

        missing = [c for c in test_chars if c not in self.characteristics]

        if missing:
            raise ValueError(f"These characteristics are not in the model: {missing}")


        # Get out the part of gamma for characteristics that should be tested
        gamma_char = self.Gamma.loc[test_chars, self.latentFactorsNames].to_numpy(dtype = float)

        # Calulate the statistic 
        W_stat = np.linalg.norm(gamma_char, "fro")**2

        


        # It is little bit harder to estimate the restricted model here, because the Z matrix should not have characteristics under null
        # so I need to cut out this characteristic

        # Need to define which characterisitcs I sitll want to keep under the null
        #keep_chars = [c for c in self.characteristics if c not in test_chars]

  
        ###### I need to change here for the correct model - the restricted one


        # Again like in the alpha test (this is again generalization) we need to estimate the models for resampling the managed portfolios
        restricted_model = self._make_char_null_model(test_chars)

        # Getting the differences in R^2
        #r2_unrestricted = self._get_total_R_squared_managed()
        #r2_restricted = restricted_model._get_total_R_squared_managed()
        r2_unrestricted = self._get_total_R_squared()
        r2_restricted = restricted_model._get_total_R_squared()

        delta_r2 = r2_unrestricted - r2_restricted

        # stack the managed portfolios to the matrix T by L
        X_matrix = np.vstack([self.X[t].loc[self.characteristics].to_numpy(dtype = float) for t in self.times])

        # get the fitted parts 
        restricted_model_fit = self._managed_fit_matrix_characteristics(restricted_model)
        unrestricted_model_fit = self._managed_fit_matrix_characteristics(self)

        # Get the residuals from the unrestricted model
        RESID = X_matrix - unrestricted_model_fit

        # Setting rng for replication
        rng = np.random.default_rng(seed)
 
        # darwing random indicies adn q's for WB or the Depednet Wild bootstrap
        boot_indices = np.empty((B, self.T), dtype = int)
        boot_q = np.empty((B, self.T), dtype = float)

        if mode == "WB":


            
            
            for b in range(B):
                boot_indices[b, :] = rng.integers(low=0, high=self.T, size=self.T)
                boot_q[b, :] = rng.standard_t(df=5, size=self.T) / np.sqrt(5.0 / 3.0)

        elif mode == "DWB":

            boot_indices = None
            ell = 6

            index = np.arange(self.T)
            distance = np.abs(index[:, None] - index[None, :])
            Sigma = np.maximum(1.0 - distance / ell, 0.0)


            for b in range(B):
                boot_q[b, :] = self._draw_bartlett_multiplier(T = self.T, rng = rng, ell = ell, Sigma = Sigma)


        # Save main results
        W_boot = np.empty(B, dtype=float)

        if useParallel:

            if printUpdates:
                print(f"Running alpha bootstrap in parallel with B = {B}, n_jobs = {n_jobs} and mode = {mode}")

            results = Parallel(
                n_jobs = n_jobs, 
                backend = "loky"
            )(
                delayed(self._single_char_bootstrap)(test_chars = test_chars, restricted_model = restricted_model, restricted_model_fit = restricted_model_fit,
                                                     RESID = RESID, mode = mode, sampled_indicies = None if boot_indices is None else boot_indices[b, :], 
                                                     q = boot_q[b, :], max_iter = max_iter, tol = tol) for b in range(B)
            )
            W_boot[:] = np.array(results, dtype = float)

        else:


            for b in range(B):
                W_boot[b] = self._single_char_bootstrap(test_chars = test_chars, restricted_model = restricted_model, restricted_model_fit = restricted_model_fit,
                                                        RESID = RESID, mode = mode, sampled_indicies = None if boot_indices is None else boot_indices[b, :],
                                                         q = boot_q[b, :], max_iter = max_iter, tol = tol )
                
                if printUpdates and ((b + 1) % max(1, B//10) == 0 or b == B - 1):
                    running_p = float(np.mean(W_boot[:b + 1] >= W_stat))

                    print(f"Bootstrap {b + 1:>5}/{B}")
                    print(f"Running p-value: {running_p:.6f}")

        # Compute the p-value of the test
        p_value = float(np.mean(W_boot >= W_stat))





        return p_value, 100 * delta_r2
    




    def _make_char_null_model(self, test_chars):
        """
        Resmapling the model under null for charactersitics test.
        """


        null_model = self.__class__(Z = self.Z, R = self.R, X = self.X, K = self.K, G = None, alpha = False)

        null_model.Gamma = self.Gamma.copy()
        null_model.F = self.F.copy()
        null_model.Lambda = self.Lambda.copy() if self.Lambda is not None else None

        null_model.Gamma.loc[test_chars, null_model.latentFactorsNames] = 0.0

        null_model.estimated = True
        null_model.converged = self.converged
        null_model.n_iter = self.n_iter
        null_model.final_tol = self.final_tol


        return null_model
    

    

    
    

    @staticmethod
    def _managed_fit_matrix_characteristics(model):

        """
        Returns a matrix T by L of the managed fit for the charactersitics equation
        """

        # Get the gamma for the multiplication
        Gamma = model.Gamma.loc[model.characteristics, model.factorNames].to_numpy(dtype = float)
        
        #Initializing the managed portfolio for the fitted part
        fitted = np.empty((model.T, model.L), dtype = float)

        for i, t in enumerate(model.times):
            W_t = model.sigmaZ[t].loc[model.characteristics, model.characteristics].to_numpy(dtype = float)

            factor_parts = []

            if model.hasLatent:
                factor_parts.append(model.F.loc[model.latentFactorsNames, t])

            if model.hasObserved:
                factor_parts.append(model.G.loc[model.observedFactorNames, t])

            f_t = pd.concat(factor_parts).loc[model.factorNames].to_numpy(dtype = float)

            fitted[i, :] = W_t.dot(Gamma).dot(f_t)

        

        return fitted
    

    def _single_char_bootstrap(self, test_chars, restricted_model, restricted_model_fit, RESID, mode, sampled_indicies, q, max_iter, tol):
        
        """
        Makes a single bootstarap for characteritics test.
        """

        if mode == "WB":
            X_boot_matrix = self._resample_single_boot_char_WB(restricted_model_fit = restricted_model_fit, RESID = RESID,
                                                                sampled_indicies = sampled_indicies, q = q)
        elif mode == "DWB":
            X_boot_matrix = self._resample_single_boot_char_DWB(restricted_model_fit = restricted_model_fit, RESID = RESID, q = q)
        else:
            raise RuntimeError("There are only two models allowed, error for single bootstrap iteration.")
        
        # Convert the created X into dictinary for the model to be estimated
        X_boot = {t: pd.Series(X_boot_matrix[i, :], index = self.characteristics, name = t) for i, t in enumerate(self.times)}

        # Make again the bootstrap model but here strictly for the characteristics
        boot_model = self.__class__(Z = self.Z, R = None, X = X_boot, K = self.K, G = None, alpha = False)

        # Get the starting values to make ti all quicker
        Gamma0 = pd.DataFrame(0.0, index = self.characteristics, columns = boot_model.factorNames)
        Gamma0.loc[:, boot_model.latentFactorsNames] = (restricted_model.Gamma.loc[self.characteristics, restricted_model.latentFactorsNames].to_numpy(dtype = float))
        F0 = restricted_model.F.loc[restricted_model.latentFactorsNames, self.times].copy()

        # Estimate the model and take out the estimated gamma to compute the statistic from a single bootstrap iteration
        Gamma_b, _, _, _, _ = self._run_unrestricted_from_restricted_start(model = boot_model, Gamma_start = Gamma0, F_start = F0, 
                                                                           max_iter = max_iter, tol = tol)
        
        # Compute the statistic
        char_stat = Gamma_b.loc[test_chars, boot_model.latentFactorsNames].to_numpy(dtype = float)
        W_b = float(np.sum(char_stat ** 2))


        
        return W_b
    

    @staticmethod
    def _resample_single_boot_char_WB(restricted_model_fit, RESID, sampled_indicies, q):
        """
        Resamples single bootstrap for WB for characteristics test.
        
        """
        return restricted_model_fit + RESID[sampled_indicies, :] * q[:, None]

    @staticmethod
    def _resample_single_boot_char_DWB(restricted_model_fit, RESID, q):
        
        """
        Resmaples single bootstrap for the DWB characteristis test.
        """
        return restricted_model_fit + RESID * q[:, None]

   










    
    
   
    

    
    
  
   





    def get_total_r_squared_by_state(self, state_labels = None):

        '''
        Returns the total R^2 based on state
        '''

        if not self.estimated:
            raise ValueError("Cannot call the function on unestimated model.")
        
        if state_labels is None:
            raise ValueError("Function needs state labels, otherwise it is not able to differenitatie between states.")
        
        #
        state_labels = pd.Series(state_labels).copy()
        state_labels.index = pd.to_datetime(state_labels.index)

        ssr = {"high": 0.0, "average": 0.0, "low": 0.0}
        sst = {"high": 0.0, "average": 0.0, "low": 0.0}

        Gamma = self.Gamma.loc[:, self.factorNames].to_numpy(float)

        for t in self.times:

            time = pd.Timestamp(t)

            state = str(state_labels.loc[time]).lower()
            Z_t = self.Z[t]
            R_t = self.R[t]

            factor_parts = []

            if self.hasLatent:
                factor_parts.append(self.F.loc[self.latentFactorsNames, t])

            if self.hasObserved:
                factor_parts.append(self.G.loc[self.observedFactorNames, t])

            f_t = pd.concat(factor_parts).loc[self.factorNames].to_numpy(float)

            fitted = Z_t.to_numpy(float).dot(Gamma).dot(f_t)
            residual = R_t.to_numpy(float) - fitted

            ssr[state] += residual.dot(residual)
            sst[state] += R_t.to_numpy(float).dot(R_t.to_numpy(float))

        high = 1.0 - ssr["high"] / sst["high"] if sst["high"] > 0 else np.nan
        average = 1.0 - ssr["average"] / sst["average"] if sst["average"] > 0 else np.nan
        low = 1.0 - ssr["low"] / sst["low"] if sst["low"] > 0 else np.nan



        return high, average, low

    def get_predictive_r_squaredby_state(self, state_labels=None):

        '''
        Returns the predictive R^2 based on state
        '''

        if not self.estimated:
            raise ValueError("Cannot call the function on unestimated model.")
        
        if state_labels is None:
            raise ValueError("Function needs state labels, otherwise it is not able to differenitatie between states.")
        
        #
        state_labels = pd.Series(state_labels).copy()
        state_labels.index = pd.to_datetime(state_labels.index)

        ssr = {"high": 0.0, "average": 0.0, "low": 0.0}
        sst = {"high": 0.0, "average": 0.0, "low": 0.0}

        Gamma = self.Gamma.loc[:, self.factorNames].to_numpy(float)
        Lambda = self.Lambda.loc[self.factorNames].to_numpy(float)

        for t in self.times:

            time = pd.Timestamp(t)

            state = str(state_labels.loc[time]).lower()
            Z_t = self.Z[t]
            R_t = self.R[t]

            fitted = Z_t.to_numpy(float).dot(Gamma).dot(Lambda)
            residual = R_t.to_numpy(float) - fitted

            ssr[state] += residual.dot(residual)
            sst[state] += R_t.to_numpy(float).dot(R_t.to_numpy(float))

        high = 1.0 - ssr["high"] / sst["high"] if sst["high"] > 0 else np.nan
        average = 1.0 - ssr["average"] / sst["average"] if sst["average"] > 0 else np.nan
        low = 1.0 - ssr["low"] / sst["low"] if sst["low"] > 0 else np.nan

        return high, average, low







    # from here on it is the out of sample exericise - to be implemented


    def fit_on_subset(self, dates, tol = 1e-6, max_iter = 1000, printTime = False, printInformation = False, Gamma0 = None):
        """
        Estimates the IPCA model on the subset of the time periods provided as a training window. 

        Primarly, used in the recursive estimation. 

        Parameters
        ----------
        dates : list
            List of time peirods (from self.times) that form a training/ estimation window.
        tol : float
            ALS convergence tolerance.
        max_iter : int
            Maximum ALS iterations, before stoptting the algorithm.
        printTime : bool
            Whether to print time in the fit iterations.
        printInformation : bool
            Whether to print diagnostic information in the fit iterations.
        Gamma0 : panda DataFrame or None
            It is the starting value for Gamma matrix.

        Returns:
        -------

        rec_model :
            Estimated IPCA model for the subset of time periods provided in ``dates``. Primarly further used for the
            forecasting out of sample.

        Raises
        ------
        ValueError
            Is raised when the method is called on the model with empty individual assets returns. It is only available, for 
            individual stocks and not managed portfolios.
        
        """

        # Need to subset the estimation data - for now only on individual assets, bc I do not think I need managed portfolios only 
        Z_subset = {t: self.Z[t] for t in dates}
        R_subset = {t: self.R[t] for t in dates} if self.R is not None else None

        # throw an error if its empty
        if R_subset is None:
            raise ValueError("This method is only implemented for the individual assets.")
        
        # Subsetting the observed factors 
        if self.hasObserved:
            # if there is alpha in the model I will cut it for now, bc thats how kelly implemented in the table 5
            # might want to develop that after I finish thesis for the package improvement - like an option for both I guess?
            #factors = [n for n in self.observedFactorNames if n != "alpha"]

            G_subset = self.G.loc[:, dates]
        else:
            G_subset = None

        rec_model = self.__class__(Z = Z_subset, R = R_subset, K = self.K, G = G_subset, alpha = self.isUnrestricted)

        rec_model.fit(tol = tol, max_iter = max_iter, printTime = printTime, printInformation = printInformation, Gamma0 = Gamma0)

        return rec_model


    def recursive_forecast(self, initial_window = 60, evaluation_window = 120, tol = 1e-6, max_iter = 1000, printUpdates = True):
        """
        Recursive out of sample estimation and forecasting.

        Estimates reursively on expanding window, and makes monthly forecasts.

        Parameters
        ----------
        initial_window : int
            Minumum number of peridos before the first forecast. Used as the first estimation.
        evaluation_window : int
            Window that is exluded from the statistics caluclation, following the implementation of Kelly et al. (2019)
        tol : float
            ALS tolerance for the fit on subset.
        max_iter : int
            ALS maximum iterations for fit on subset.
        printUpdates : bool
            Prints progress after each estimation. So, monthly updates.

        Returns
        -------
        results : dictionary with keys
            total_R2 - total R^2
            predictive_R2 - predictive R^2
            oos_dates - dates with out-of-sample prediction
            squared_errors_total - dictionary with arrays of per asset squared errors total for each time period as the key
            squared_errors_predictive - dictionary with arrays of per asset squared errors predictive for each time period as the key
            factor_returns - dictionary with all factor returns series for each time period as the key
            lambda_series - dictionary with predictive returns of all factors (series) for each time period as the key.

        Raises
        ------
        ValueError
            Raises when the forecast is called on managed portfolios only. This implementation is only for the individual assets right now.
        ValueError
            Raises when the initial window is longer, than the whole dataset provided to the model.
        ValueError
            Raises when the evaluation window is longer, than the whole dataset provided to the model.
        

        """

        # what if I would implement here a rolling window? maybe would perform better than Kelly? if I have time, I would possibly implement that
        # if not also for potential package expanding after I finish thsi thesis


        # make the check s
        if self.R is None:
            raise ValueError("This method requires individual asset returns.")
        
        whole_dataset = self.times
        T = len(self.times)

        if initial_window >= T:
            raise ValueError("The initial window is longer than the model dataset.")
        if evaluation_window >= T:
            raise ValueError("The evaltiation window excludes more observations than are provided.")


        forecasts_no = T - initial_window



        oos_total_ssr = 0.0
        oos_predictive_ssr = 0.0
        oos_sst = 0.0


        oos_managed_total_ssr = 0.0
        oos_managed_predictive_ssr = 0.0
        oos_managed_sst = 0.00
      


        oos_dates = []
        evaluation_dates = []

        sq_err_total = {}
        sq_err_predictive = {}
        factor_returns = {}
        lambda_series = {}

        prev_Gamma = None

        
        #main forecast loop
        for i in range(initial_window, T):

            t_next = whole_dataset[i]
            t_prev = whole_dataset[i - 1]

            # Expand the traiing dataset
            train_dates = whole_dataset[:i]


            # Estimating the training model 
            trained_model = self.fit_on_subset( dates = train_dates, tol = tol, max_iter = max_iter, Gamma0 = prev_Gamma)

            prev_Gamma = trained_model.Gamma

            Gamma_latent = trained_model.Gamma.loc[self.characteristics, trained_model.latentFactorsNames].to_numpy(float)
            Gamma_full = trained_model.Gamma.loc[self.characteristics, trained_model.factorNames].to_numpy(float)

            # need the running window lambda for the predictive R^2
            lambda_latent = trained_model.F.loc[trained_model.latentFactorsNames, :].mean( axis = 1).to_numpy(float)

            if trained_model.hasObserved:

                obs_real = [n for n in trained_model.observedFactorNames if n != "alpha"]
                if obs_real:
                    lambda_observed = trained_model.G.loc[obs_real, :].mean(axis = 1).to_numpy(float)
                    lambda_full = np.concatenate([lambda_latent, lambda_observed])
                else:
                    lambda_full = lambda_latent

                
            else:
                lambda_full = lambda_latent

            
            # Creatiung the forecasted returns with the r_t+1

            sigmaZ_t = self.sigmaZ[t_next].values
            x_next = self.X[t_next].values

            # instead of inverstion, again like i did in the ALS algorithm, I solve for f with the OLS - it is just quicker 
            numerator_f = Gamma_latent.T.dot(x_next)
            denominator_f = Gamma_latent.T.dot(sigmaZ_t).dot(Gamma_latent)

            # actually solve it 
            f_pred = np.linalg.lstsq(denominator_f, numerator_f, rcond = None)[0]

            if trained_model.hasObserved:
                obs_real = [n for n in trained_model.observedFactorNames if n != "alpha"]
                
                if obs_real:
                    g_next = self.G.loc[obs_real, t_next].to_numpy(float)
                    f_full = np.concatenate([f_pred, g_next])
                else:
                    f_full = f_pred


                
            else:
                f_full = f_pred

            # sacing the predictions
            factor_returns[t_next] = f_full
            lambda_series[t_next] = lambda_full

            # clauclate the statisitsc only for stocks that actually have Z_t and r_t+1 
            common_stocks = self.Z[t_next].index.intersection(self.R[t_next].index)
            Z_mat = self.Z[t_next].loc[common_stocks].to_numpy(float)
            R_vec = self.R[t_next].loc[common_stocks].to_numpy(float)


            # Calcuate the errors 
            error_total = R_vec - Z_mat.dot(Gamma_full).dot(f_full)
            error_predictive = R_vec - Z_mat.dot(Gamma_full).dot(lambda_full)

            oos_dates.append(t_next)

            sq_err_total[t_next] = error_total ** 2
            sq_err_predictive[t_next] = error_predictive ** 2


            if i >= evaluation_window:

                ## calcualte the R^2 components and add them up in the loop
                ssr_total = float(error_total.dot(error_total))
                ssr_predictive = float(error_predictive.dot(error_predictive))
                sst = float(R_vec.dot(R_vec))

                oos_total_ssr += ssr_total
                oos_predictive_ssr += ssr_predictive
                oos_sst += sst
                evaluation_dates.append(t_next)

                ## Added this bc i actually want at the end the manged portoflio model git measures
                managed_fit_total = sigmaZ_t.dot(Gamma_full).dot(f_full)
                managed_fit_pred =  sigmaZ_t.dot(Gamma_full).dot(lambda_full)


                managed_error_total = x_next - managed_fit_total
                managed_error_pred = x_next - managed_fit_pred

                oos_managed_total_ssr += float(managed_error_total.dot(managed_error_total))
                oos_managed_predictive_ssr += float(managed_error_pred.dot(managed_error_pred))
                oos_managed_sst += float(x_next.dot(x_next))



            if printUpdates and (i == 1 or i % 20 == 0):
                print(f"Finished {i - initial_window + 1} out of {forecasts_no} forecasts.")
        

        oos_total_r2 = 1.0 - oos_total_ssr / oos_sst
        oos_predictive_r2 = 1.0 - oos_predictive_ssr / oos_sst

        oos_managed_total_r2 = 1.0 - oos_managed_total_ssr / oos_managed_sst
        oos_managed_pred_r2 = 1.0 - oos_managed_predictive_ssr / oos_managed_sst

    

        if printUpdates:
            print(f" OOS total R2: {100 * oos_total_r2}")
            print(f" OOS predictive R2: {100 * oos_predictive_r2}")
        
        results = {"total_R2": oos_total_r2, "predictive_R2": oos_predictive_r2, "oos_dates": oos_dates,
                    "managed_total_R2": oos_managed_total_r2, "managed_predictive_R2": oos_managed_pred_r2,
                     "squared_errors_total": sq_err_total, "evaluation_dates": evaluation_dates,
                    "squared_errors_predictive": sq_err_predictive, "factor_returns": factor_returns, "lambda_series": lambda_series}

        return results



    def diebold_mariano_test(errors_a, errors_b):

        """
        Diebold-Mariano test using the precomputed squared forecasts errors.

        The DM test based on Diebold, F. X. and Mariano, R. S. (2002) paper, with the two sided alternative.


        Parameters
        ----------
        errors_a : dictionary
            Squared forecast errors of model A, where keys are time periods and values are the squared forecast errors.
        errors_b : dictionary
            Squared forecast errors of model B, where keys are time periods and values are the squared forecast errors.

        Returns
        -------
        results : dictionary with keys
            dm_stat - Diebold-Mariano statistic
            p_value - test p-value
            mean_loss_diff - mean loss differential between the two loss functions
            se - corresponding stadard error
            n - number of predictions.
        
        Raises
        ------
        ValueError
            If the passed erros of model A and model B are for different periods.

        """


        # check for the periods contained
        keys_a = set(errors_a.keys())
        keys_b = set(errors_b.keys())

        # throw and error if the error peridos do not mathc 
        if keys_a != keys_b:
            raise ValueError("Errors of model A and B should have the same length and contain the same error periods.")
        
        keys = sorted(keys_a)

        se1 = np.asarray([errors_a[k] for k in keys], dtype = float)
        se2 = np.asarray([errors_b[k] for k in keys], dtype = float)
        
        
        
        # compute the loss differential
        d = se1 - se2

        X = np.ones((len(d), 1))

        res = sm.OLS(d, X).fit(cov_type = "HAC", cov_kwds = {"maxlags": 0 }, use_t = False)

        dm_stat = float(res.tvalues[0])
        p_value = float(2 * (1 - norm.cdf(abs(dm_stat))))

        results = {"dm_stat": dm_stat, "p_value": p_value, "mean_loss_diff": float(res.params[0]), "se": float(res.bse[0]), "n": len(d)}

        return results














    
        







    











    

    
    
    



















   


    def _get_residuals(self):

        '''
        Get the estimated residuals.

        This is an old function and it is actually no longer needed. I leave it in case I would need in the future package expansions.
        '''

        # Checking the required inputs for the testing alpha pricing error test (the test already in baseline Kelly)

        if not self.estimated:
            raise ValueError("The model needs to be first estimated before the residuals can be computed.")
        
        if self.R is None:
            raise ValueError("Assetr returns R are needed to compute the asset-level residuals.")
        
        # Creating the initial dictionaries for the two types of residuals: epsilon and d (in Kelly and mine notation)
        asset_residuals = {}
        d = {}

        # Makeing sure you get correct gamma
        Gamma = self.Gamma.loc[:, self.factorNames].values

        # Build the residual for each time t in self times
        for t in self.times:

            factors_included = []

            if self.hasLatent:
                factors_included.append(self.F.loc[self.latentFactorsNames, t])
            
            if self.hasObserved:
                factors_included.append(self.G.loc[self.observedFactorNames, t])

            all_included_f = pd.concat(factors_included).loc[self.factorNames].values

            # Get the object of of dics for the time we are in now

            Z_t = self.Z[t]
            R_t = self.R[t]

            fitted = Z_t.values.dot(Gamma).dot(all_included_f)

            eps_t = pd.Series(R_t.values -fitted, index = R_t.index, name = t)

            asset_residuals[t] = eps_t

            # gett the d's residuals
            d_t = Z_t.T.dot(eps_t) #/ self.N_t[t]
            d_t.name = t

            d[t] = d_t

        return asset_residuals, d





























    
    






