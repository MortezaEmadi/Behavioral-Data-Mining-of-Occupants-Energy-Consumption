"""
All Rights Reserved
@author Morteza Emadi
This module will get all of the rules mined by FPGrowth or some selected ones and try to
implement a MLP for each of them and if it meet the evaluation metrics the results will be saved
for each rule info and its corresponding potential saving. Lastly, it will save the cumulative results
of all the calculated ANNs of a home
#Note: For PLOTTING a selected rule go to line 780
"""

import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import KFold
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
import re
import warnings
from data_preprocess import homeid
from sklearn.metrics import r2_score
#####** ToDo!! : note: ((( * data with "Tune" names here refer to *Validation* data  )))

def Standardize(X):

    meanX = np.mean(X,axis=0)
    stdX = np.std(X,axis=0)
    zeros = np.argwhere(stdX == 0)
    stdX[zeros] = 1
    ##for cte coumns
    # stdX[stdX==0] = 1
    X -= meanX
    X /= stdX
    return meanX,stdX,X


def real_normalize(X,min_x=None,max_x=None):
    if min_x is None:
        max_x = X.max(axis=0)
        min_x = X.min(axis=0)

    diff_x = max_x - min_x
    zeros = np.argwhere(diff_x == 0)
    if zeros.size:
        diff_x[zeros] = 1
    normal_x = (X - min_x) / diff_x
    return max_x, min_x, normal_x

def denormalize(x, min_x, max_x):

    return x * (max_x - min_x) + min_x

def Destandardize(X,meanX,stdX):

    return (X * stdX) + meanX

def RMSE(y,pred_x):
    return np.sqrt(np.mean((pred_x - y) ** 2))
    # return (np.mean((regr.predict(x)-y)**2))**0.5


def fullmlp(limiter, my_homeid):
    results = []
    rulecounter = 0
    total_rules = len(limiter)
    current_file_summary = {}
    for fileName in limiter:
        rulecounter += 1
        print(f"{round((rulecounter/len(limiter))*100,2)}%= rule {rulecounter} of {total_rules} rules is loading! & {len(limiter) - total_rules} rules from tot of {len(limiter)}were deleted till now")
        # fileName = csvFiles[0]
        homeid = fileName.split("_")[0][4:]
        if (fileName[-8:] != "_org.csv") or (str(my_homeid) not in homeid):
            rulecounter -= 1
            total_rules -= 1

            continue
        ruleStr = fileName.split("_")[1]

        print('-------------',ruleStr)
        csvFile = basePath + fileName
        data = pd.read_csv(csvFile)

        dataAnti = pd.read_csv(basePath+"home"+str(homeid)+"_"+str(ruleStr)+'_anti.csv')

        inputColumns = []
        outputColumns = []
        for col in data.columns:
            if col[:5]=="cons_":
                outputColumns.append(col)
            if col[:5]=="antc_":
                inputColumns.append(col)

        X = data[inputColumns]
        Y = data[outputColumns]

    #######for testing our code,with a fake dataset!!==>
        # X, Y = make_regression(n_features=2, n_samples=70, random_state=0)

        max_x, min_x, X = real_normalize(np.array(X))
        max_y, min_y, Y = real_normalize(np.array(Y))

        ################### Normal Antiiiis!==>
        xAnti = np.array(dataAnti[inputColumns])
        yAnti = np.array(dataAnti[outputColumns])

        ###########!!!!!!!!!!!  Normalize VS Standardize=
        # xAnti = (xAnti-meanX)/stdX
        # yAnti = ((yAnti-meanY)/stdY).reshape([-1])

        max_anti_x, min_anti_x, xAnti = real_normalize(xAnti, min_x, max_x)


        max_anti_y, min_anti_y, yAnti = real_normalize(yAnti, min_y, max_y)
        yAnti = yAnti.reshape([-1])
        a = pd.DataFrame(X)
        non_zeros = a.loc[~(a == 0).all(axis=1)].index
        X = np.array(X[non_zeros])
        zero_column= np.argwhere(np.all(X[..., :] == 0, axis=0))
        X = np.delete(X, zero_column, axis=1)
        xAnti = np.delete(xAnti, zero_column, axis=1)
        if X.shape[0] < 2:
            rulecounter -= 1
            total_rules -=1
            print(f"rule {ruleStr} is deleted!")
            continue

        Y = np.array(Y[non_zeros])



        if (xAnti.size < anti_min_mlp_prep) or (X.size < org_min_mlp_prep):
            rulecounter -= 1
            total_rules -= 1
            print(f"rule {ruleStr} is deleted since after omitting ALL-ZEROS occurences there were no/few occurences of it(ORG/Anti) left!")
            continue


        Y = Y.reshape([-1])
        X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(X) ,pd.Series(Y) ,  test_size=0.2, random_state=1)

        result_df = pd.DataFrame(index=X_test.index, columns=['Random','Naive_Mean','LR','Actual'])
        result_df.Actual = y_test

        # Method: Naive Mean
        result_df.Naive_Mean = y_train.mean()

        # Method: Random
        result_df.Random = np.random.uniform(y_train.min(), y_train.max(),y_test.shape)

        #Method: LR
        car_lm = LinearRegression()
        car_lm.fit(X_train, y_train)
        result_df.LR =car_lm.predict(X_test)


        result_df['|Random-Actual|'] = abs(result_df.Random-result_df.Actual)
        result_df['|Naive_Mean-Actual|'] = abs(result_df.Naive_Mean-result_df.Actual)
        result_df['|LR-Actual|'] = abs(result_df.LR-result_df.Actual)

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        metric_df_none = pd.DataFrame(index = ['ME','RMSE', 'MAE','MAPE'] ,
                                 columns = ['Random','Naive_Mean','LR'])

        n_test = len(result_df)

        for m in metric_df_none.columns:
            metric_df_none.at['ME',m]= np.sum((result_df.Actual - result_df[m]))/n_test
            metric_df_none.at['RMSE',m]= np.sqrt(np.sum(result_df.apply(lambda r: (r.Actual - r[m])**2,axis=1))/n_test)
            metric_df_none.at['MAE',m] = np.sum(abs(result_df.Actual - result_df[m]))/n_test
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metric_df_none.at['MAPE',m] = np.sum(result_df.apply(lambda r:abs(r.Actual-r[m])/r.Actual,axis=1))/n_test*100
        metric_df_none

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #### MLP
        car_mlp = MLPRegressor(hidden_layer_sizes=my_hls,max_iter=my_iter)
        car_mlp.fit(X_train, y_train)

        result_df['MLP'] = car_mlp.predict(X_test)
        result_df['|MLP-Actual|'] = abs(result_df.MLP-result_df.Actual)
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        m='MLP'

        metric_df_none.at['ME',m] = np.sum((result_df.Actual - result_df[m]))/n_test
        metric_df_none.at['RMSE',m] = np.sqrt(np.sum(result_df.apply(lambda r: (r.Actual - r[m])**2,axis=1))/n_test)
        metric_df_none.at['MAE',m] = np.sum(abs(result_df.Actual - result_df[m]))/n_test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metric_df_none.at['MAPE',m] = np.sum(result_df.apply(lambda r:abs(r.Actual-r[m])/r.Actual,axis=1))/n_test*100

        metric_df_none


        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        ####Todo!....>>> Done! I have done this very precisely by splitting a test part from all the data and catch the
        # tune and test of each kf-split from just the (Whole-Test) part!

        ## important parametre:
        num_repetition = 3
        kf = KFold(n_splits=num_repetition, shuffle=True)
        X_train_s, X_tune, y_train_s, y_tune = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        ###########            Tune for activation and solver

        # Create a placeholder for experimentations
        activation_options = ['identity', 'logistic', 'tanh', 'relu']
        solver_options = ['lbfgs','sgd','adam']


        my_index = pd.MultiIndex.from_product([activation_options,solver_options],
                                             names=('activation', 'solver'))

        tune_df_activ_solver = pd.DataFrame(index = my_index,
                               columns=['R{}'.format(i) for i in range(num_repetition)])

        tune_df_activ_solver

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        n = len(y_tune)
        for activation_o in activation_options:
            for solver_o in solver_options:
                # for rep in tune_df_activ_solver.columns:
                ii = -1
                for trainIdx, testIdx in kf.split(X_train):
                    ii += 1
                    rep = tune_df_activ_solver.columns[ii]
                    X_train_s = X_train.iloc[trainIdx]
                    y_train_s = y_train.iloc[trainIdx]
                    X_tune = X_train.iloc[testIdx]
                    y_tune = y_train.iloc[testIdx]

                    # regr = clone(model).fit(trainX, trainY)
                    # scores.append(RMSE(regr.predict(testX), testY))
                    car_mlp = MLPRegressor(hidden_layer_sizes=(my_hls), max_iter=my_iter,
                                           activation=activation_o, solver=solver_o)

                    car_mlp.fit(X_train_s, y_train_s)
                    y_tune_predict = car_mlp.predict(X_tune)
                    RSME = np.sqrt(np.sum((y_tune_predict - y_tune) ** 2) / n)

                    tune_df_activ_solver.at[(activation_o, solver_o), rep] = RSME

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        tune_df_activ_solver['Mean'] = tune_df_activ_solver[
            ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
        tune_df_activ_solver['Min'] = tune_df_activ_solver[
            ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
        tune_df_activ_solver = tune_df_activ_solver.sort_values('Mean')
        activation_tuned = tune_df_activ_solver.index[0][0]
        solver_tuned = tune_df_activ_solver.index[0][1]
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        ### Tune for hidden_layer_sizes
        # This code is basically creating a list of 15 one layered ANN ([1] - [15])
        # and 100 two layered ANNs ([1,1] - [10,10])
        PossibleNetStrct = []
        PossibleNetString = []

        for i in range(3, 5):
            netStruct = [(5*i)+2]
            PossibleNetStrct.append(netStruct)
            PossibleNetString.append(str(netStruct))

        for i in [5,6,10,13]:
            for j in [5,9,14,19,20]:
                netStruct = [i, j]
                PossibleNetStrct.append(netStruct)
                PossibleNetString.append(str(netStruct))

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        tune_df_layer = pd.DataFrame(np.nan, index=PossibleNetString,
                               columns=['R{}'.format(i) for i in range(num_repetition)])

        n = len(y_tune)

        for i, netStr in enumerate(PossibleNetStrct):
            RowName = PossibleNetString[i]
            # for rep in tune_df_layer.columns:
            ii = -1
            for trainIdx, testIdx in kf.split(X_train):
                ii += 1
                rep = tune_df_layer.columns[ii]
                X_train_s = X_train.iloc[trainIdx]
                y_train_s = y_train.iloc[trainIdx]
                X_tune = X_train.iloc[testIdx]
                y_tune = y_train.iloc[testIdx]
                car_mlp = MLPRegressor(hidden_layer_sizes=netStr, activation=activation_tuned,
                                       solver=solver_tuned, max_iter=my_iter)
                car_mlp.fit(X_train_s, y_train_s)

                y_tune_predict = car_mlp.predict(X_tune)
                RSME = np.sqrt(np.sum((y_tune_predict - y_tune) ** 2) / n)

                tune_df_layer.at[RowName, rep] = RSME
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        tune_df_layer['Mean'] = tune_df_layer[
            ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
        tune_df_layer['Min'] = tune_df_layer[
            ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)

        tune_df_layer = tune_df_layer.sort_values('Mean')

        layer_tuned = tuple([int(i) for i in tune_df_layer.index[0][1:-1].split(",")])
        ######
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        ### Tune for learning_rate, learning_rate_init
        LR_options = ['constant','invscaling','adaptive']
        LRI_options = [0.0001,0.001,0.005,0.01,0.05,0.1]

        my_index = pd.MultiIndex.from_product([LR_options,LRI_options],
                                             names=('LR', 'rate'))

        tune_df_lrn_lrni = pd.DataFrame(index = my_index,
                               columns=['R{}'.format(i) for i in range(num_repetition)])

        tune_df_lrn_lrni
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        for LR_o in LR_options:
            for LRI_o in LRI_options:
                # for rep in tune_df_lrn_lrni.columns:
                ii = -1
                for trainIdx, testIdx in kf.split(X_train):
                    ii += 1
                    rep = tune_df_lrn_lrni.columns[ii]
                    X_train_s = X_train.iloc[trainIdx]
                    y_train_s = y_train.iloc[trainIdx]
                    X_tune = X_train.iloc[testIdx]
                    y_tune = y_train.iloc[testIdx]
                    car_mlp = MLPRegressor(hidden_layer_sizes=layer_tuned, max_iter=my_iter,
                                           activation=activation_tuned, solver=solver_tuned,
                                           learning_rate=LR_o, learning_rate_init=LRI_o)

                    car_mlp.fit(X_train_s, y_train_s)
                    y_tune_predict = car_mlp.predict(X_tune)
                    RSME = np.sqrt(np.sum((y_tune_predict - y_tune) ** 2) / n)

                    tune_df_lrn_lrni.at[(LR_o, LRI_o), rep] = RSME
                # print((LR_o, LRI_o))
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #

        tune_df_lrn_lrni['Mean'] = tune_df_lrn_lrni[
            ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
        tune_df_lrn_lrni['Min'] = tune_df_lrn_lrni[
            ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
        tune_df_lrn_lrni = tune_df_lrn_lrni.sort_values('Mean')
        learningrate_tuned = tune_df_lrn_lrni.index[0][0]
        learning_initrate_tuned = tune_df_lrn_lrni.index[0][1]
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        ## Tune for max_iter, shuffle

        max_iterations_options = [1000,2000]
        shuffle_options = [True,False]

        my_index = pd.MultiIndex.from_product([max_iterations_options,shuffle_options],
                                             names=('Max Iterations', 'shuffle'))

        tune_df_maxitr_shuf = pd.DataFrame(index = my_index,
                               columns=['R{}'.format(i) for i in range(num_repetition)])

        tune_df_maxitr_shuf

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        for max_iterations_o in max_iterations_options:
            for shuffle_o in shuffle_options:
                # for rep in tune_df_maxitr_shuf.columns:
                ii = -1
                for trainIdx, testIdx in kf.split(X_train):
                    ii += 1
                    rep = tune_df_maxitr_shuf.columns[ii]
                    X_train_s = X_train.iloc[trainIdx]
                    y_train_s = y_train.iloc[trainIdx]
                    X_tune = X_train.iloc[testIdx]
                    y_tune = y_train.iloc[testIdx]
                    car_mlp = MLPRegressor(hidden_layer_sizes=layer_tuned, max_iter=my_iter,
                                           activation=activation_tuned, solver=solver_tuned,
                                           learning_rate=learningrate_tuned, learning_rate_init=learning_initrate_tuned,
                                           shuffle=shuffle_o)

                    car_mlp.fit(X_train_s, y_train_s)
                    y_tune_predict = car_mlp.predict(X_tune)
                    RSME = np.sqrt(np.sum((y_tune_predict - y_tune) ** 2) / n)

                    tune_df_maxitr_shuf.at[(max_iterations_o, shuffle_o), rep] = RSME
                # print((max_iterations_o, shuffle_o))
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        tune_df_maxitr_shuf['Mean'] = tune_df_maxitr_shuf[
            ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
        tune_df_maxitr_shuf['Min'] = tune_df_maxitr_shuf[
            ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)
        tune_df_maxitr  = tune_df_maxitr_shuf.sort_values('Mean')
        maxiter_tuned = tune_df_maxitr.index[0][0]
        shuffle_tuned = tune_df_maxitr.index[0][1]

        alpha_tuned = 0.00005

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        # Tune for Randomness
        random_options = [12,32]

        tune_df_rndm = pd.DataFrame(index=random_options,
                               columns=['R{}'.format(i) for i in range(num_repetition)])
        tune_df_r2 = pd.DataFrame(index=random_options,
                                    columns=['R{}'.format(i) for i in range(num_repetition)])
        # tune_df_rndm
        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================

        for random_o in random_options:
            # for rep in tune_df_rndm.columns:
            ii = -1
            for trainIdx, testIdx in kf.split(X_train):
                ii += 1
                rep = tune_df_rndm.columns[ii]
                X_train_s = X_train.iloc[trainIdx]
                y_train_s = y_train.iloc[trainIdx]
                X_tune = X_train.iloc[testIdx]
                y_tune = y_train.iloc[testIdx]

                #########################
                car_mlp = MLPRegressor(hidden_layer_sizes=layer_tuned, max_iter=maxiter_tuned,
                                       activation=activation_tuned, solver=solver_tuned, learning_rate=learningrate_tuned,
                                       learning_rate_init=learning_initrate_tuned, shuffle=shuffle_tuned, alpha=alpha_tuned, random_state=random_o)
                car_mlp.fit(X_train_s, y_train_s)
                y_tune_predict = car_mlp.predict(X_tune)
                RSME = np.sqrt(np.sum((y_tune_predict - y_tune) ** 2) / n)
                r2 = r2_score(y_tune, y_tune_predict)
                ##deghat kon in loop ham dare cv behine ro baraye rsme_MIN ejra mikone va ham vase
                # random dare behine peida mikone,vali vase r2 ma faghat mean of it in kfold ro mikhaym!hamin!
                tune_df_rndm.at[random_o, rep] = RSME
                tune_df_r2.at[random_o, rep] = r2
            # print(random_o)
        # ======================================================================================

        # ======================================================================================
        tune_df_rndm['Mean'] = tune_df_rndm[
            ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
        tune_df_rndm['Min'] = tune_df_rndm[
            ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)

        tune_df_r2['Mean'] = tune_df_r2[
            ['R{}'.format(i) for i in range(num_repetition)]].mean(axis=1)
        tune_df_r2['Min'] = tune_df_r2[
            ['R{}'.format(i) for i in range(num_repetition)]].min(axis=1)

        tune_df_rndm = tune_df_rndm.sort_values('Mean')
        random_tuned = tune_df_rndm.sort_values('Mean').index[0]

        tune_df_r2 = tune_df_r2.sort_values('Mean')
        # r2_cv =

    ##########################################################################################################
                                    # Finalizing
        ###########################################################################################
        ### ** !! semi!-Finally here: (best of hyperparams yeilding to)CV error:
        cv_error = tune_df_rndm.iloc[0]["Mean"]
        r2_cv = tune_df_r2.iloc[0]["Mean"]

        #  *************  # Train the tuned MLP on train set
        car_mlp = MLPRegressor(hidden_layer_sizes=layer_tuned, max_iter=maxiter_tuned,
                               activation=activation_tuned, solver=solver_tuned, learning_rate=learningrate_tuned,
                               learning_rate_init=learning_initrate_tuned, shuffle=shuffle_tuned, alpha=alpha_tuned, random_state=random_tuned)

        car_mlp.fit(X_train, y_train)

        y_test_predict = car_mlp.predict(X_test)
        RMSE_test = np.sqrt(np.sum((y_test_predict - y_test) ** 2) / n)
        r2_test = r2_score(y_test, y_test_predict)


        result_df['MLP_tuned'] = car_mlp.predict(X_test)
        result_df['|MLP_tuned-Actual|'] = abs(result_df.MLP_tuned - result_df.Actual)

        table = result_df[['|Random-Actual|', '|Naive_Mean-Actual|', '|LR-Actual|', '|MLP-Actual|', '|MLP_tuned-Actual|']]

        # ======================================================================================================================================
        #             #########################End of Cell ######################################
        # ======================================================================================================================================
        #
        m = 'MLP_tuned'

        metric_df_none.at['ME', m] = np.sum((result_df.Actual - result_df[m])) / n_test
        metric_df_none.at['RMSE', m] = np.sqrt(np.sum(result_df.apply(lambda r: (r.Actual - r[m]) ** 2, axis=1)) / n_test)
        metric_df_none.at['MAE', m] = np.sum(abs(result_df.Actual - result_df[m])) / n_test

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metric_df_none.at['MAPE', m] = np.sum(result_df.apply(lambda r: abs(r.Actual - r[m]) / r.Actual, axis=1)) / n_test * 100


        if (RMSE_test > rmse_test_threshold) or (r2_test < r2_test_fail):
            print(f"Error on MLP!RMSE_test={RMSE_test},r2_test={r2_test} Rule {ruleStr} is deleted! Since RSME_test and r2_test were awful!")
            rulecounter -= 1
            total_rules -= 1
            continue
        print(f"going to Antirules 'coz RMSE_test={RMSE_test},r2_test={r2_test} Rule {ruleStr}!")
    #************************************************************************************************
        # =============================== #End of MLP tuning and testing!###############################
    #************************************* Start working with AntiRules ************************ =

        predAnti = car_mlp.predict(xAnti)

        dataAnti['pred'] = denormalize(predAnti, min_y, max_y)




        yAnti = pd.DataFrame(denormalize(yAnti, min_y, max_y), index=dataAnti.index)[0]

        if ruleStr[-2] == "1":
            dataAnti['potentialsaving'] = dataAnti.pred - yAnti
            kind = "reinforcing"
        else:  #####cons_ORG=0
            dataAnti['potentialsaving'] = yAnti - dataAnti.pred
            kind = "correction"

        if int(re.findall("(?<=\s)(\d*)", ruleStr)[-1][:-1]) > 10:
            energykind = "Gas"
        else:
            energykind = "Electricity"

        ##if potential is unacceptable..this is good for choosing "maenadar" rules and also choosing for plots
        if len(dataAnti[dataAnti["potentialsaving"] < 0]) > 0:
            num_bdantis = len(dataAnti[dataAnti["potentialsaving"] < 0])
            dataAnti.loc[dataAnti["potentialsaving"] < 0, ["potentialsaving"]] = 0.000001
            if (num_bdantis / len(dataAnti)) > 0.5 or (len(dataAnti) - num_bdantis) < 8:
                print(f"Rule {ruleStr} is deleted!      Since many of its antis' potential savings were negative")
                rulecounter -= 1
                total_rules -= 1
                continue
            else:
                print(f"some potential savings of rule {ruleStr} were invalid")
        else:
            num_bdantis = 0

        tot_rule_saving = round(dataAnti['potentialsaving'].sum(), 6)

        ###time stamp of first and last
        first_Anti_occur = dataAnti["time"].values[0]
        last_Anti_occur = dataAnti["time"].values[-1]

        if ruleStr == plot_rule:
            ##Todo: garche behtar bood ru khode test neshun midadi ta shak nakonand overfit shode,ama eb nadare ala!=
            plotdata = pd.DataFrame()
            plotdata['pred'] = denormalize(car_mlp.predict(X_train),min_y, max_y)
            # data['pred'] = Destandardize(predOrg, meanY, stdY)
            plotdata.loc[plotdata.pred < 0, 'pred'] = 0
            plotdata["ytrain"] = denormalize(y_train,min_y, max_y)

            plotdata.to_csv(resultPath + re.findall("(.*?)(?=\])", fileName)[0] + "]" + '_org_pred_4plot.csv')
            z = 700
            plt.plot(plotdata.pred[z:z + 100], label="ANN Output")
            plt.plot(plotdata.ytrain[z:z + 100], label="True Values", alpha=0.7)
            # plt.plot(data.pred[90:], label="ANN Output")
            # plt.plot(data[outputColumns][90:], label="True Values", alpha=0.7)
            plt.legend(loc='best', fontsize=14)
            plt.title(f'Training ANN for rule: {plot_rule} in home {my_homeid}', fontsize=14)

            plt.xlabel("Instances of Rule Occurrences", fontsize=14)
            plt.ylabel("Consequents Value (KWh)", fontsize=14)
            plt.show()
            plt.close()
            startindex = 0
            endindex = -1
            yy = {6: "Mar", 10: "Apr", 11: "Apr", 12: "Apr", 1: "May", 2: "Jun", 3: "non", 4: "noon", 9: "nooon"}

            dataAnti.time = dataAnti.time.apply(
                lambda x: x[8:10] + " " + yy[int(x[5:7])] + " " + x[11: -3])
            nullindex = dataAnti[(dataAnti.pred < 0) | (dataAnti[outputColumns].squeeze() < dataAnti.pred)].index
            dataAnti.drop(dataAnti.index[nullindex], inplace=True)
            # acceptableindexes=dataAnti.index
            labels = dataAnti.time[startindex:endindex]
            g1 = dataAnti.pred[startindex:endindex].apply(lambda x: round(x, 2))
            g2 = dataAnti[outputColumns].squeeze()[startindex:endindex].apply(lambda x: round(x, 2))

            x = np.arange(labels.shape[0])
            width = 0.35

            fig, ax = plt.subplots()
            rects1 = ax.bar(x - width / 2, g1, width, label='ANN Prediction')
            rects2 = ax.bar(x + width / 2, g2, width, label='Disobedience Value')

            ax.set_xticks(x, labels, rotation=72)

            ax.bar_label(rects1, padding=3, )
            ax.bar_label(rects2, padding=3)

            ax.legend(loc='best', fontsize=12)

            fig.tight_layout()
            plt.title(f'AntiRule Occurrences and their corresponding ANN Predictions for rule {plot_rule} in home 145',
                      fontsize=12)
            plt.xlabel("Timestamps of AntiRule Occurrences", fontsize=12)
            plt.ylabel("Energy Values (KWh)", fontsize=12)

            plt.show()
            plt.close()
            #######      ================ =======================================================================================================
            ##This is the number of datapoints which is desired but from the end of the series(nerar 2018 spring)
            num_plot = 60

            cons_data = data[outputColumns[0]]
            # threshold = np.percentile(cons_data, 75)
            # cons_data[cons_data > threshold] = 0
            data_normal_cons = real_normalize(cons_data)

            nullindex_cons = data_normal_cons[(data_normal_cons.values >= 0.18)].index

            for col in inputColumns:

                antc_dataa = data[col]

                if col == 'antc_energy_kettle_5210':

                    threshold = np.percentile(antc_dataa, 99)

                else:

                    threshold = np.percentile(antc_dataa, 75)

                antc_dataa[antc_dataa > threshold] = 0

                if "radiator" in col:
                    data_normal_intc1 = real_normalize(antc_dataa)
                    nullindex_intc1 = data_normal_intc1[(data_normal_intc1.values >= 0.2)].index
                else:
                    data_normal_intc2 = real_normalize(antc_dataa)
                    nullindex_intc2 = data_normal_intc2[(data_normal_intc2.values <= 0.072)].index

            all_nullindex = list(set(list(nullindex_cons) + list(nullindex_intc1) + list(nullindex_intc2)))
            all_nullindex = sorted(all_nullindex)
            data_normal_intc1.drop(data_normal_intc1.index[list(all_nullindex)], inplace=True)
            data_normal_intc2.drop(data_normal_intc2.index[list(all_nullindex)], inplace=True)
            data_normal_cons.drop(data_normal_cons.index[list(all_nullindex)], inplace=True)
            data_normal_intc1.reset_index(drop=True, inplace=True)
            data_normal_intc2.reset_index(drop=True, inplace=True)
            data_normal_cons.reset_index(drop=True, inplace=True)
            plt.plot(data_normal_intc1[-num_plot:],
                     label=f"antc 190={inputColumns[1].split('_')[2]}_{inputColumns[1].split('_')[3]}", marker="o")
            plt.plot(data_normal_intc2[-num_plot:], label=f"antc 41={inputColumns[0].split('_')[2]}", marker="o")
            plt.plot(data_normal_cons[-num_plot:], label=f"cons 60={outputColumns[0].split('_')[2]}", marker="o")
            plt.legend(loc='best', fontsize=10)
            plt.title(f'Depiction of relation among  {plot_rule} rule\'s readings in home {homeid}')
            plt.xlabel("Instances of Rule Occurrences")
            plt.ylabel("Power Values (KW)")

            plt.show()
            break
        if (RMSE_test < rmse_test_acc) and (r2_test > r2_test_acc) and (energykind == "Gas"):
            current_file_summary.update({'Gas_valid_potentialsaving': tot_rule_saving})
        if (RMSE_test < rmse_test_acc) and (r2_test > r2_test_acc) and (energykind == "Electricity"):
            current_file_summary.update({'Electric_valid_potentialsaving': tot_rule_saving})
        dataAnti.to_csv(resultPath + re.findall("(.*?)(?=\])", fileName)[0] + "]" + '_anti_pred.csv')

        current_file_summary = {'rule': ruleStr, 'cv_rmse': round(cv_error, 6), 'rmse': round(RMSE_test, 6),
                                'cv_r2': round(r2_cv, 6), 'r2_test': round(r2_test, 6),
                                "hidden_layer": layer_tuned, "max_iter": maxiter_tuned,
                                "activation": activation_tuned,
                                "solver": solver_tuned, "learning_rate": learningrate_tuned,
                                "learning_rate_init": learning_initrate_tuned,
                                "shuffle": shuffle_tuned, "alpha": alpha_tuned, "random_state": random_tuned,
                                "Size_Rules": len(data),
                                "Size_AntiRules": len(dataAnti), "size_Sp_Antirules": num_bdantis,
                                "1st_Antitime": first_Anti_occur,
                                "last_Antitime": last_Anti_occur, "Rule_kind": kind, "Energy_kind": energykind,
                                }

        if (energykind == "Gas"):
            current_file_summary.update({'Gas_potentialsaving': tot_rule_saving})
            current_file_summary.update({"Gas_saving_percentageof_antirules": tot_rule_saving / (yAnti.sum())})
        else:
            current_file_summary.update({'Electric_potentialsaving': tot_rule_saving})
            current_file_summary.update({"Electric_saving_percentageof_antirules": tot_rule_saving / (yAnti.sum())})

    results.append(current_file_summary)
    results = pd.DataFrame(results)
    try:
        results['home_Electric_potential_save'] = [np.sum(results['Electric_potentialsaving'])] + [''] * (
                    len(results) - 1)
    except:
        pass

    try:
        results['home_valid_Electric_potential_save'] = [np.sum(results['Electric_valid_potentialsaving'])] + [''] * (
                len(results) - 1)
    except:
        pass
    try:
        results['home_Gas_potential_save'] = [np.sum(results['Gas_potentialsaving'])] + [''] * (len(results) - 1)
    except:
        pass

    try:
        results['home_valid_Gas_potential_save'] = [np.sum(results['Gas_valid_potentialsaving'])] + [''] * (
                len(results) - 1)
    except:
        pass

    results.to_csv(resultPath + f'FinalResult{my_homeid}.csv')


if __name__ == '__main__':
    ##setting initial hyperparamethers:
    my_hls = 20
    my_actv = 'relu'
    my_slvr = 'lbfgs'
    my_iter = 2000
    my_alph = 0.00005
    my_lr = 'constant'
    my_lrint = 0.001
    my_shuf = True
    ####################################
    rmse_test_threshold = 0.1
    rmse_test_acc = 0.001
    r2_test_fail = -1
    r2_test_acc = 0.4
    # -----------------------------------------
    org_min_before_mlp = 600  ##### dar annver2 Module in ro bas eslah koni
    org_min_mlp_prep = 500
    # -----------------------------
    anti_min_before_mlp = 12  ##### dar annver2 Module in ro bas eslah koni
    anti_min_mlp_prep = 6
    #======================================================================
    homeid=homeid
    with open("directory_temp.txt", "r") as file_temp:
        basePath = file_temp.readline()
    ########It is prefered to change this location whenever you run this module, so as to have distinct answers
    resultPath = './finalann_1401/' + str(homeid) + "_" + str(7) + "/"
    _, _, csvFiles = next(os.walk(basePath))
    results = []
    my_homeid = homeid

    #Todo: Important Note:
    # IF you want just examine particular rules put them in front of "selected_rule =" &
    # & if you want to plot a particular rule just put it in front of "plot_rule ="
    # & if you don't specify none of above options it will examine all the rules in the basePath

    # Example: selected_rule = "[14651, 70, 14641]"
    selected_rule = ""  # As above line, do not forget the spaces after each comma!
    plot_rule = ""


    if selected_rule:
        limiter = [x for x in csvFiles if selected_rule in x and "org" in x]  # does a exist in the current namespace
    elif plot_rule:
        limiter = [x for x in csvFiles if plot_rule in x and "org" in x]
    else:
        limiter = [x for x in csvFiles if "org" in x]
    fullmlp(limiter,my_homeid)