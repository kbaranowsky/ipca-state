"""
Data preperation for the Instrumented Principal Component Analysis (IPCA) and State Conditioned 
Instrumented Principal Component Analsysi (SC-IPCA) Estimators.

Prepares the raw asset level panel dataset from Kelly et al. (2019) into the inputs required by the IPCA estimator.
The main outputs are dictionaries of characteristics pandas DataFrames, and returns pandas Series.

The preparation incluces date normalization, excess returns construction, cross-sectional rank 
standarization, filtering months and assets and state interaction construction for the state
conditioned IPCA estimator.

Adtionally, it perpares state interacted characteristics, according to Baranowski, K. (2026) methodology.

Main class
----------
Instruments
    Prepare asset level return and characterisitcs data for IPCA and SC-IPCA estimation

    
References
----------

Baranowski, K. (2026). State Conditioning in Instrumented Principal
Component Analysis: Do Market States Change Characteristics Based Risk
Exposures? Unpublished Bachelor thesis, Erasmus School of Economics, Erasmus University Rotterdam.

Kelly, B. T., Pruitt, S., and Su, Y. (2019). Characteristics are covariances: A unified model of
risk and return. Journal of Financial Economics, 134(3):501–524.   

"""



import numpy as np
import pandas as pd


class Instruments:

    """

    Prepare asset level panel data and state variable for the estimation of IPCA and SC-IPCA.

    The class transofmrs the raw asset level panel dataset from Kelly et al. (2019) and the 
    state varaible into the inputs required by the IPCA estimator. For each time priod, the data
    constists of dataframe of asset characterisics and the series of different asset returns. 

    Args: 
        data : pandas DataFrame
            Raw asset level panel dataset as provieded by Kelly et al. (2019). It must contain date, 
            asset identifier, returns and characteristics columsn.
        characteristics : list of str
            Column names in ``data`` that contains the asset characterisitcs.
        returns : str
            Column name in ``data`` that contains asset returns.
        date : str
            Column name in ``data`` that contains the dates of observations.
        permno : str
            Column name in ``data`` that contains the asset identifiers (in the Kelly et al. (2019) dataset these are permnos).
        risk_free : pandas dataframe or None
            Two column dataframe containing dates (first column) and risk-free rate (second column) in decimals. If passed to the method
            it is subtructed from the asset returns, to convert them to excess returns.
        filterMonths : bool, default True
            Whether to filter the months where there is minimum number of assets in that month. (rn it is 100 - maybe change that for general)
        state_variable : pandas dataFrame, default None
            Two column dataframe containing dates (first column) and state variable (second column) in decimals. It is required, when
            making the interacted characteristics for SC-IPCA.
        isFirstRowName : bool, default False
            Whether the first row of ``data`` contaisn the column names.



    Attributes:
        data : pandas dataframe
            Raw asset level panel dataset from Kelly et al. (2019).
        characteristics : list of str
            Names of characteristics columms.
        returns : str
            Name of the return column.
        date : str
            Name of the date column.
        permno : str
            Name of the asset identifier colmn.
        risk_free : pandas dataframe, or None
            Risk-free rate panda dataframe.
        state_variable : pandas dataframe,or None
            State variable series data.
        isFirstRowName : bool
            Indicator for whether the first row in the raw dataset is column names.
        Z : dictionary or None
            Dictionary with characteristics matricies, where keys are time periods.
        R : dictionary or None
            Dictionary of returns series, where keys are time periods.
        numOfTime : int or None
            NUmber of time periods in the perpared dataset.
        numOfAssets : int or None
            Number of unique assets in the prepared dataset.
        numOfChara : int or None
            Number of characterisitcs in the prepared dataset, inclduing the constant if created.


    Notes:
        Characteristics are cross-sectionally transofrmed, as described in the Kelly et al. (2019)

    
    """

    def __init__ (self, data, characteristics, returns, date, permno, risk_free = None, filterMonths = True, state_variable = None, isFirstRowName = False):

        """

        Initializes the Instrument object, based on passed parameters.

        Parameters
        -----------
        data : pandas dataframe
            Raw asset level panel dataset of Kelly et al. (2019).
        characteristics : list of str
            Column names containing the asset characteristics.
        returns : str
            Column name with assets returns.
        date : str
            Column name iwth dates.
        permno : str
            Column names with asset identifiers.
        risk_free : pandas dataframe, default None
            Two column dataframe with dates and risk-free rate.
        filterMonths : bool, default True
            Whether to filter months accoridng to the rule where at least 100 assets are reuqired.
        state_variable : pandas dataframe, default None
            Dataframe with date and state variable columns.
        isFirstRowName : bool, default False
            Whether the first row in the raw daraset contains the column names.

        """

        self.data = data.copy()
        self.characteristics = characteristics
        self.returns = returns
        self.state_variable = state_variable
        self.date = date
        self.permno = permno
        self.risk_free = risk_free
        self.isFirstRowName = isFirstRowName

        # Outputs
        self.Z = None
        self.R = None
        #self.S = None
        self.numOfTime = None
        self.numOfAssets = None
        self.numOfChara = None
        

    def prepare_data(self, addConstant = True, filterMonths = True, make_state_interactions = False, filterStocks = False, min_months = None, printSummary = False):
        
        '''

        This function prepares the data for the IPCA estimation. It creatres ready objects for the IPCA class.

        It also allows for the state dependent variable standarization, and therfore can prepare the data
        for the state conditioned IPCA extension.

        Parameters
        -----------
        addConstant : bool, default True
            If true, it adds a constant to a characteristics column. In line with the Kelly et al. (2019).
        filterMonths : bool, default True
            Whether to filter months, for these with at least 100 assets.
        make_state_interactions : bool, default False
            Whether to make the characteristics intercated with the state variable.
        filterStocks : bool, default False
            Whether to keep assets with at least ``min_months`` in the dataset.
        min_months : int, defualt None
            Minimum number of months that an asset needs, when filter according to that.
        printSummary : bool, default False
            If true it prints summaryu information about the perpared dataset.

        Returns
        --------
        Z : dictionary
            Dictionary with characteristics dataframes, where keys are time peridos.
        R : dictionary
            Dictionary with asset returns series, where keys are itme periods.
        
        Raises
        -----
        ValueError
            If required columns are missing in the raw dataset.
        ValueError
            If iteractions are requested, but no state variable is provided.

        '''

        if make_state_interactions is True and self.state_variable is None:
            raise ValueError ("Cannot make state interaction if the instrument class was not passed the ready state variable!")
        


        # Make the dataframe standard such that the first row is already data

        if self.isFirstRowName:
            self.data.columns = self.data.iloc[0]
            self.data = self.data[1:].reset_index(drop = True)
            self.data.columns.name = None



        # Check if the columns from characteristics and returns are in the data 
        required_columns = [self.date, self.permno, self.returns] + self.characteristics

        missing_columns = [
            col for col in required_columns
            if col not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        #if standardize_state_variable == True:
        #    if self.state_variable is None:
        #       raise ValueError("Cannot standardize the state variable, becuase it is missing.")
            

        # Standardize the dates to make sure they are months end to be able to comapre later with the state variable
        self.data[self.date] = pd.to_datetime(self.data[self.date]) + pd.offsets.MonthEnd(0)
            
        # adding the cutting options, for when I estimate the table 2 with the restriction of at least 60 months for stocks incluedeed
        if filterStocks:
            self._filterStocks(min_months = min_months)

        # apply the authros filter in the nuber of assets a month and the rank condition 
        if filterMonths:
            self._filterMonths(addConstant = addConstant)


        if make_state_interactions:

            state = self.state_variable.copy()

            state_date_col = state.columns[0]
            state_value_col = state.columns[1]

            state[state_date_col] = pd.to_datetime(state[state_date_col]) + pd.offsets.MonthEnd(0)

            state = (
                state[[state_date_col, state_value_col]]
                .dropna()
                .drop_duplicates(subset=state_date_col, keep="last")
                .rename(columns={
                    state_date_col: "_state_date",
                    state_value_col: "_state"
                })
            )
                    
            self.data = self.data.merge(
                state, 
                left_on=self.date, 
                right_on="_state_date", 
                how="inner"
            )

            self.data = self.data.drop(columns=["_state_date"])



        

        # Adding the interactions

            

        # Cross standardize the characteristics
        self.data[self.characteristics] = (
            self.data
            .groupby(self.date)[self.characteristics]
            .transform(self._rank_standardize_to_kelly_interval)
        )
        
        if make_state_interactions:
            self.data[self.characteristics] = self.data[self.characteristics].mul(self.data["_state"], axis = 0)


        # Add a constant to the characteristics
        if addConstant:
            constant_name = "constant"
            z_columns = [constant_name] + self.characteristics
        else: 
            z_columns = self.characteristics

        # Create the dictionaries 
        self.R = {}
        self.Z = {}

        for date, group in self.data.groupby(self.date):
            group = group.sort_values(self.permno)
            group = group.set_index(self.permno)

            group[constant_name] = 1.0

            self.R[date] = group[self.returns]
            self.Z[date] = group[z_columns]


        # Subtruct the risk free rate 
        unique_dates = self.data[self.date].unique()

        if self.risk_free is not None:

            rf_date_col = self.risk_free.columns[0]
            rf_value_col = self.risk_free.columns[1]

            self.risk_free[rf_date_col] = pd.to_datetime(self.risk_free[rf_date_col]) + pd.offsets.MonthEnd(0)
            self.risk_free = self.risk_free.set_index(rf_date_col)[rf_value_col]

            for date in unique_dates:
                self.R[date] = self.R[date] - self.risk_free[date]
        

        # Standardize the state variable 

        #if standardize_state_variable:
        #    self.S = self._standardize_state(self.state_variable)
        # else:
        #    self.S = self.state_variable

        
        # Adding here summary statistics information
        self.numOfTime = len(self.Z)
        self.numOfAssets = self.data[self.permno].nunique()
        self.numOfChara = len(z_columns)


        # Printing the summary statistics

        if printSummary:
            print("=" * 70)
            print("Prepared dataset summary")
            print("=" * 70)

            print(f"Number of time periods:                                  {self.numOfTime}")
            print(f"Number of unique assets:                                {self.numOfAssets}")
            print(f"Number of characteristics (including constant if added): {self.numOfChara}")
            print(f"Constant added to characteristics:                       {addConstant}")
            print(f"Risk-free rate subtructed from returns:                  {self.risk_free is not None}")
            if make_state_interactions:
                print(f"Prepared state interactions:                           {make_state_interactions}")
            #print(f"State variable included:                                 {self.S is not None}")
            #print(f"State variabel standardized:                             {standardize_state_variable}")


        return self.Z, self.R


    @staticmethod
    def _rank_standardize_to_kelly_interval(x):
        '''
        Cross sectional rank standarization to the intercal [-0.5, 0.5].

        Parameters
        ----------
        x : pandas series
            Cross section of one characteristics for one time period.
        
        Returns
        ---------
        out : pandas series
            Rank standaardzied series.
        
        '''

        n_t = x.notna().sum()

        if n_t <= 1:
            return x * np.nan
        
        ranks = x.rank(method = "average", na_option = "keep")

        out = (ranks - 1) / (n_t - 1) - 0.5

        return out
    


    def _filterMonths(self, addConstant = True, min_cross_section_assets = 100):

        '''
        Filters months with insufficient number of assets.

        Parameters
        ----------
        addConstant : bool, default True
            Whether the characteristcsi include the constant.
        min_cross_section_assets : int, defualt 100
            Minimum of assets required per month.

        '''

        L = len(self.characteristics)

        # Authros thresdhold adjusted bc I am not going to use the measn thing
        threshold = max(min_cross_section_assets, L + int(addConstant) + 1)


        # Veryfing for missing observations - I need non missing charactersitics anf retunrs

        non_missing = (self.data[self.characteristics].notna().all(axis = 1) & self.data[self.returns].notna())
        n_valid_by_row_month = non_missing.groupby(self.data[self.date]).transform("sum")

        # Keep onlyt those that satisfy the restrictions
        keep = non_missing & (n_valid_by_row_month >= threshold)

        self.data = self.data.loc[keep].copy().reset_index(drop = True)


    def _filterStocks(self, min_months = 12):

        """

        Filters assets for these that have little returns history.

        Parameters
        ----------
        min_months : int, default 12
            Minumum number of months that asset has to have to not be filtered.

        """

       
        valid = self.data[self.returns].notna()

        obs_count = (self.data.loc[valid].groupby(self.permno)[self.date].nunique())
        #obs_count = self.data.groupby(self.permno)[self.date].nunique()

        keep_assets = obs_count[obs_count >= min_months].index

        #self.data = (self.data.loc[valid & self.data[self.permno].isin(keep_assets)].copy().reset_index(drop = True))
        self.data = self.data.loc[self.data[self.permno].isin(keep_assets)].copy().reset_index(drop = True)




        





    @staticmethod
    def _standardize_state(x):

        '''
        The method standardizes the state variable. The result variable is approx mean zero and std 1.

        It is a hidden method, and right now not used. It is a leftover from the old code, but I leave it now
        for possible future package extension.

        Parameters
        ----------
        x : pandas series
            Series of state variable.
        
        Returns
        --------
        out : pandas series
            Series of standardzied state variable.
        
        Raises
        ------
        ValueError
            If the standard deviation of series is 0 or null.

        '''


        mean  = x.mean()
        sd = x.std()

        if pd.isna(sd) or sd == 0:
            raise ValueError("Cannot standardize state variable because its standard deviation is zero or missing.")

        out = (x - mean) / sd

        return out
    

    
