# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 09:54:15 2020

@author: dhulse
"""
## This file shows different data visualization of trade-off analysis of the cost models with different design variables
# like battery, rotor config, operational height at a level of resilience policy.
# The plots gives a general understanding of the design space, trade-offs between cost models (obj func.), sensitivity of
# subsystem w.r.t models, and effect of subsystem config and operational variables on different cost models

# Few examples have been provided for interpretation. However, different plotting other than shown here can be done depending
# on the analysis question or for better visualization.
import sys
sys.path.append('../../')

import fmdtools.faultsim.propagate as propagate
import fmdtools.resultdisp as rd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns; sns.set(style="ticks", color_codes=True)
# from drone_mdl import *
# import time
# from drone_opt import *
# import pandas as pd
# import numpy as np
#
# # Design Model
# xdes1 = [0, 1]
# desC1 = x_to_dcost(xdes1)
# print(desC1)
#
# # Operational Model
# xoper1 = [122] #in m or ft?
# desO1 = x_to_ocost(xdes1, xoper1)
# print(desO1)
#
# #Resilience Model
# xres1 = [0, 0]
# desR1 = x_to_rcost(xdes1, xoper1, xres1)
# print(desR1)
#
# #all-in-one model
# xdes1 = [3,2]
# xoper1 = [65]
# xres1 = [0,0]
#
# a,b,c,d = x_to_ocost(xdes1, xoper1)
#
# mdl = x_to_mdl([0,2,100,0,0])
#
#
# endresults, resgraph, mdlhist = propagate.nominal(mdl)
#
# rd.plot.mdlhistvals(mdlhist, fxnflowvals={'StoreEE':'soc'})

# Read the dataset of cost model values and constraint validation for a large grid of design variables
grid_results= pd.read_csv('grid_results_new.csv')
#print(grid_results.head())
#print(grid_results.shape)

# Portion of feasible data among the whole dataset
feasible_DS =(grid_results['c_cum'].isin([0]).sum())/len(grid_results)
#print("The portion of feasible design space from the grid results")
#print(feasible_DS)

#Subsetting only feasible data
grid_results_FS = grid_results[(grid_results['c_cum']==0)]
g = sns.pairplot(grid_results_FS, hue="ResPolBat", vars=["Bat", "Rotor","Height","desC","operC","resC"], corner=True, diag_kind="kde",kind="reg")
plt.show()


########################## Optimization results from different framework#################################
# Optimization framework involved: Bi-level, Two-Stage and Single MOO (Weighted Tchebycheff)
opt_results= pd.read_csv('opt_results.csv')
#print(opt_results.head())
#print(opt_results.shape)
obj1 = pd.Series.tolist(opt_results['Obj1'])
obj2 = pd.Series.tolist(opt_results['Obj2'])
index= ['Bi-LevelP1000', 'Bi-LevelP100', 'Bi-LevelP10/1', 'Two-Stage', 'MOO:w1=0','MOO:w1=[0.1,0.2,0.3]','MOO:w1=0.4','MOO:w1=[0.5,0.6,..,1]']
df_y = pd.DataFrame({'Obj1:DesC+OperC':obj1, 'Obj2:FailureC': obj2}, index=index)
df_y.plot.bar(rot=45)
plt.title("Costs at optimal decision under different frameworks")
plt.show()

obj_combined = pd.Series.tolist(opt_results['Obj1']+opt_results['Obj2'])
df_y1 = pd.DataFrame({'Obj1 + Obj2:DesC+OperC+FailureC':obj_combined}, index=index)
df_y1.plot.bar(rot=45)
plt.title("Combined Costs at optimal decision under different frameworks")
plt.show()
# #Subsetting all data into 4 parts- each resilience policy (includes also infeasible data)
# # grid_results_res0 = grid_results[grid_results['ResPolBat']==0] #continue
# # grid_results_res1 = grid_results[grid_results['ResPolBat']==1] #to_home
# # grid_results_res2 = grid_results[grid_results['ResPolBat']==2] #to_nearest
# # grid_results_res3 = grid_results[grid_results['ResPolBat']==3] #emland
#
# #Subsetting only feasible data
# grid_results_res0 = grid_results[(grid_results['ResPolBat']==0) & (grid_results['c_cum']==0)] #continue
# grid_results_res1 = grid_results[(grid_results['ResPolBat']==1) & (grid_results['c_cum']==0)] #to_home
# grid_results_res2 = grid_results[(grid_results['ResPolBat']==2) & (grid_results['c_cum']==0)] #to_nearest
# grid_results_res3 = grid_results[(grid_results['ResPolBat']==3) & (grid_results['c_cum']==0)] #emland
#
# # print(grid_results_res0.shape)
# # print(grid_results_res1.shape)
# # print(grid_results_res2.shape)
# # print(grid_results_res3.shape)
#
# grid_results_res0_mh = grid_results_res0.groupby(['Bat', 'Rotor']).mean()
# grid_results_res1_mh = grid_results_res1.groupby(['Bat', 'Rotor']).mean()
# grid_results_res2_mh = grid_results_res2.groupby(['Bat', 'Rotor']).mean()
# grid_results_res3_mh = grid_results_res3.groupby(['Bat', 'Rotor']).mean()
#
# grid_results_res0_mh1 = grid_results_res0.groupby(['Bat']).mean()
# grid_results_res0_mh2 = grid_results_res0.groupby(['Rotor']).mean()
# grid_results_res1_mh1 = grid_results_res1.groupby(['Bat']).mean()
# grid_results_res1_mh2 = grid_results_res1.groupby(['Rotor']).mean()
# grid_results_res2_mh1 = grid_results_res2.groupby(['Bat']).mean()
# grid_results_res2_mh2 = grid_results_res2.groupby(['Rotor']).mean()
# grid_results_res3_mh1 = grid_results_res3.groupby(['Bat']).mean()
# grid_results_res3_mh2 = grid_results_res3.groupby(['Rotor']).mean()
#
# plt.clf()
#
# #############################################################################################
# # The first section of plot is to show the trend of design, operation and failure cost models at different bat/rotor
# # config at a fixed level of resilience policy, at mean height after ignoring the infeasible design choices.
#
# # We plot with the normalized cost values for better visualization
# x=pd.Series(np.arange(1, len(grid_results_res0_mh)+1, 1))
# my_xticks = ['MQ','MH','MO','SQ','SH','SO','PQ','PH','PO','BQ','BH','BO']
# #Normalizing the cost value
# plt.subplot(2,1,1)
# y1=(grid_results_res0_mh['desC']-grid_results_res0_mh['desC'].min())/(grid_results_res0_mh['desC'].max()-grid_results_res0_mh['desC'].min())
# y2=(grid_results_res0_mh['operC']-grid_results_res0_mh['operC'].min())/(grid_results_res0_mh['operC'].max()-grid_results_res0_mh['operC'].min())
# y3=(grid_results_res0_mh['resC']-grid_results_res0_mh['resC'].min())/(grid_results_res0_mh['resC'].max()-grid_results_res0_mh['resC'].min())
# plt.xticks(x, my_xticks)
# plt.plot(x,y1,'*b-')
# plt.plot(x,y2,'or-')
# plt.plot(x,y3,'.g-')
# plt.xlabel("Bat/Rot Config")
# plt.ylabel("Cost")
# plt.legend(('Des','Oper','Res'))
# plt.title("Config vs Cost on policy: cont.")
# #plt.show()
#
# plt.subplot(2,1,2)
# y1=(grid_results_res1_mh['desC']-grid_results_res1_mh['desC'].min())/(grid_results_res1_mh['desC'].max()-grid_results_res1_mh['desC'].min())
# y2=(grid_results_res1_mh['operC']-grid_results_res1_mh['operC'].min())/(grid_results_res1_mh['operC'].max()-grid_results_res1_mh['operC'].min())
# y3=(grid_results_res1_mh['resC']-grid_results_res1_mh['resC'].min())/(grid_results_res1_mh['resC'].max()-grid_results_res1_mh['resC'].min())
# plt.xticks(x, my_xticks)
# plt.plot(x,y1,'*b-')
# plt.plot(x,y2,'or-')
# plt.plot(x,y3,'.g-')
# plt.xlabel("Bat/Rot Config")
# plt.ylabel("Cost")
# plt.legend(('Des','Oper','Res'))
# plt.title("Config vs Cost on policy: to_home")
# plt.show()
#
# plt.subplot(2,1,1)
# y1=(grid_results_res2_mh['desC']-grid_results_res2_mh['desC'].min())/(grid_results_res2_mh['desC'].max()-grid_results_res2_mh['desC'].min())
# y2=(grid_results_res2_mh['operC']-grid_results_res2_mh['operC'].min())/(grid_results_res2_mh['operC'].max()-grid_results_res2_mh['operC'].min())
# y3=(grid_results_res2_mh['resC']-grid_results_res2_mh['resC'].min())/(grid_results_res2_mh['resC'].max()-grid_results_res2_mh['resC'].min())
# plt.xticks(x, my_xticks)
# plt.plot(x,y1,'*b-')
# plt.plot(x,y2,'or-')
# plt.plot(x,y3,'.g-')
# plt.xlabel("Bat/Rot Config")
# plt.ylabel("Cost")
# plt.legend(('Des','Oper','Res'))
# plt.title("Config vs Cost on policy: to_nearest")
# #plt.show()
#
# plt.subplot(2,1,2)
# y1=(grid_results_res3_mh['desC']-grid_results_res3_mh['desC'].min())/(grid_results_res3_mh['desC'].max()-grid_results_res3_mh['desC'].min())
# y2=(grid_results_res3_mh['operC']-grid_results_res3_mh['operC'].min())/(grid_results_res3_mh['operC'].max()-grid_results_res3_mh['operC'].min())
# y3=(grid_results_res3_mh['resC']-grid_results_res3_mh['resC'].min())/(grid_results_res3_mh['resC'].max()-grid_results_res3_mh['resC'].min())
# plt.xticks(x, my_xticks)
# plt.plot(x,y1,'*b-')
# plt.plot(x,y2,'or-')
# plt.plot(x,y3,'.g-')
# plt.xlabel("Bat/Rot Config")
# plt.ylabel("Cost")
# plt.legend(('Des','Oper','Res'))
# plt.title("Config vs Cost on policy: emland")
# plt.show()
#
#
# #################################################################
# # The second section is to plot bar graphs to find the sensitivity between battery and rotor w.r.t Cost functions at a
# # fixed level of resilience policy
# #plt.subplot(2,1,1)
#
# y1=(grid_results_res0_mh1['desC'])
# y1_std = y1.std()
# y2=(grid_results_res0_mh2['desC'])
# y2_std = y2.std()
# y3=(grid_results_res1_mh1['desC'])
# y3_std = y3.std()
# y4=(grid_results_res1_mh2['desC'])
# y4_std = y4.std()
# y5=(grid_results_res2_mh1['desC'])
# y5_std = y5.std()
# y6=(grid_results_res2_mh2['desC'])
# y6_std = y6.std()
# y7=(grid_results_res3_mh1['desC'])
# y7_std = y7.std()
# y8=(grid_results_res3_mh2['desC'])
# y8_std = y8.std()
# y_bat_std = [y1_std, y3_std, y5_std, y7_std]
# y_rot_std = [y2_std, y4_std, y6_std, y8_std]
# index= ['Cont.', 'To_home', 'To_nearest', 'emland']
# df_y = pd.DataFrame({'Battery':y_bat_std, 'Rotor': y_rot_std}, index=index)
# df_y.plot.bar(rot=0)
# plt.title("Sensitivity analysis of systems: Design Cost")
# plt.show()
#
# #plt.subplot(2,1,1)
# y1=(grid_results_res0_mh1['operC'])
# y1_std = y1.std()
# y2=(grid_results_res0_mh2['operC'])
# y2_std = y2.std()
# y3=(grid_results_res1_mh1['operC'])
# y3_std = y3.std()
# y4=(grid_results_res1_mh2['operC'])
# y4_std = y4.std()
# y5=(grid_results_res2_mh1['operC'])
# y5_std = y5.std()
# y6=(grid_results_res2_mh2['operC'])
# y6_std = y6.std()
# y7=(grid_results_res3_mh1['operC'])
# y7_std = y7.std()
# y8=(grid_results_res3_mh2['operC'])
# y8_std = y8.std()
# y_bat_std = [y1_std, y3_std, y5_std, y7_std]
# y_rot_std = [y2_std, y4_std, y6_std, y8_std]
# index= ['Cont.', 'To_home', 'To_nearest', 'emland']
# df_y = pd.DataFrame({'Battery':y_bat_std, 'Rotor': y_rot_std}, index=index)
# df_y.plot.bar(rot=0)
# plt.title("Sensitivity analysis of systems: Operational Cost")
# plt.show()
#
# #plt.subplot(2,1,1)
# y1=(grid_results_res0_mh1['resC'])
# y1_std = y1.std()
# y2=(grid_results_res0_mh2['resC'])
# y2_std = y2.std()
# y3=(grid_results_res1_mh1['resC'])
# y3_std = y3.std()
# y4=(grid_results_res1_mh2['resC'])
# y4_std = y4.std()
# y5=(grid_results_res2_mh1['resC'])
# y5_std = y5.std()
# y6=(grid_results_res2_mh2['resC'])
# y6_std = y6.std()
# y7=(grid_results_res3_mh1['resC'])
# y7_std = y7.std()
# y8=(grid_results_res3_mh2['resC'])
# y8_std = y8.std()
# y_bat_std = [y1_std, y3_std, y5_std, y7_std]
# y_rot_std = [y2_std, y4_std, y6_std, y8_std]
# index= ['Cont.', 'To_home', 'To_nearest', 'emland']
# df_y = pd.DataFrame({'Battery':y_bat_std, 'Rotor': y_rot_std}, index=index)
# df_y.plot.bar(rot=0)
# plt.title("Sensitivity analysis of systems: Failure Cost")
# plt.show()
#
#
# #####################################################################
# ## The third section of the plot is to consider the height variables as well.
# # Here we plot to investigate the effect of cost values with the change of height at each feasible combination of
# # battery/rotor (at that height) for a fixed level of resilience policy
#
# # We plot with the normalized cost values for better visualization
# # Resilience policy: Continue
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res0_xdesvsxoper = grid_results_res0[(grid_results_res0['Bat'] == ibat) & (grid_results_res0['Rotor'] == irot)]
#         x= grid_results_res0_xdesvsxoper['Height']
#         y= (grid_results_res0_xdesvsxoper['desC']-grid_results_res0['desC'].min())/(grid_results_res0['desC'].max()-grid_results_res0['desC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Design Cost")
#         plt.title("Height vs Cost for each config on policy: cont.")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res0_xdesvsxoper = grid_results_res0[(grid_results_res0['Bat'] == ibat) & (grid_results_res0['Rotor'] == irot)]
#         x= grid_results_res0_xdesvsxoper['Height']
#         y= (grid_results_res0_xdesvsxoper['operC']-grid_results_res0['operC'].min())/(grid_results_res0['operC'].max()-grid_results_res0['operC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Operational Cost")
#         plt.title("Height vs Cost for each config on policy: cont.")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res0_xdesvsxoper = grid_results_res0[(grid_results_res0['Bat'] == ibat) & (grid_results_res0['Rotor'] == irot)]
#         x= grid_results_res0_xdesvsxoper['Height']
#         y= (grid_results_res0_xdesvsxoper['resC']-grid_results_res0['resC'].min())/(grid_results_res0['resC'].max()-grid_results_res0['resC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Failure Cost")
#         plt.title("Height vs Cost for each config on policy: cont.")
#
# plt.show()
#
# # Resilience Policy: to_home
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res1_xdesvsxoper = grid_results_res1[(grid_results_res1['Bat'] == ibat) & (grid_results_res1['Rotor'] == irot)]
#         x= grid_results_res1_xdesvsxoper['Height']
#         y= (grid_results_res1_xdesvsxoper['desC']-grid_results_res1['desC'].min())/(grid_results_res1['desC'].max()-grid_results_res1['desC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Design Cost")
#         plt.title("Height vs Cost for each config on policy: to home")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res1_xdesvsxoper = grid_results_res1[(grid_results_res1['Bat'] == ibat) & (grid_results_res1['Rotor'] == irot)]
#         x= grid_results_res1_xdesvsxoper['Height']
#         y= (grid_results_res1_xdesvsxoper['operC']-grid_results_res1['operC'].min())/(grid_results_res1['operC'].max()-grid_results_res1['operC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Operational Cost")
#         plt.title("Height vs Cost for each config on policy: to home")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res1_xdesvsxoper = grid_results_res1[(grid_results_res1['Bat'] == ibat) & (grid_results_res1['Rotor'] == irot)]
#         x= grid_results_res1_xdesvsxoper['Height']
#         y= (grid_results_res1_xdesvsxoper['resC']-grid_results_res1['resC'].min())/(grid_results_res1['resC'].max()-grid_results_res1['resC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Failure Cost")
#         plt.title("Height vs Cost for each config on policy: to home")
#
# plt.show()
#
#
# # Resilience Policy: to_nearest
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res2_xdesvsxoper = grid_results_res2[(grid_results_res2['Bat'] == ibat) & (grid_results_res2['Rotor'] == irot)]
#         x= grid_results_res2_xdesvsxoper['Height']
#         y= (grid_results_res2_xdesvsxoper['desC']-grid_results_res2['desC'].min())/(grid_results_res2['desC'].max()-grid_results_res2['desC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Design Cost")
#         plt.title("Height vs Cost for each config on policy: to nearest")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res2_xdesvsxoper = grid_results_res2[(grid_results_res2['Bat'] == ibat) & (grid_results_res2['Rotor'] == irot)]
#         x= grid_results_res2_xdesvsxoper['Height']
#         y= (grid_results_res2_xdesvsxoper['operC']-grid_results_res2['operC'].min())/(grid_results_res2['operC'].max()-grid_results_res2['operC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Operational Cost")
#         plt.title("Height vs Cost for each config on policy: to nearest")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res2_xdesvsxoper = grid_results_res2[(grid_results_res2['Bat'] == ibat) & (grid_results_res2['Rotor'] == irot)]
#         x= grid_results_res2_xdesvsxoper['Height']
#         y= (grid_results_res2_xdesvsxoper['resC']-grid_results_res2['resC'].min())/(grid_results_res2['resC'].max()-grid_results_res2['resC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Failure Cost")
#         plt.title("Height vs Cost for each config on policy: to nearest")
#
# plt.show()
#
#
# # Resilience Policy: emland
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res3_xdesvsxoper = grid_results_res3[(grid_results_res3['Bat'] == ibat) & (grid_results_res3['Rotor'] == irot)]
#         x= grid_results_res3_xdesvsxoper['Height']
#         y= (grid_results_res3_xdesvsxoper['desC']-grid_results_res3['desC'].min())/(grid_results_res3['desC'].max()-grid_results_res3['desC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Design Cost")
#         plt.title("Height vs Cost for each config on policy: emland")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res3_xdesvsxoper = grid_results_res3[(grid_results_res3['Bat'] == ibat) & (grid_results_res3['Rotor'] == irot)]
#         x= grid_results_res3_xdesvsxoper['Height']
#         y= (grid_results_res3_xdesvsxoper['operC']-grid_results_res3['operC'].min())/(grid_results_res3['operC'].max()-grid_results_res3['operC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Operational Cost")
#         plt.title("Height vs Cost for each config on policy: emland")
#
# plt.show()
# np.random.seed(100)
# for ibat in pd.Series([0, 1, 2, 3]):
#     for irot in pd.Series([0, 1, 2]):
#         grid_results_res3_xdesvsxoper = grid_results_res3[(grid_results_res3['Bat'] == ibat) & (grid_results_res3['Rotor'] == irot)]
#         x= grid_results_res3_xdesvsxoper['Height']
#         y= (grid_results_res3_xdesvsxoper['resC']-grid_results_res3['resC'].min())/(grid_results_res3['resC'].max()-grid_results_res3['resC'].min())
#         plt.plot(x,y,'*-', c=np.random.rand(3,))
#         plt.xlabel("Height (m)")
#         plt.ylabel("Failure Cost")
#         plt.title("Height vs Cost for each config on policy: emland")
#
# plt.show()



