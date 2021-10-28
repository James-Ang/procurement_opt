# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:41:56 2019


@author: Acceval Pte Ltd
"""
import scipy.optimize as optimize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import pandas as pd
import win32api
import datetime


""" Initialisation"""
A = 0.1/730         # WACC / (2*365 days)
OI_init = 2000      # Inventory starting Point
WAMC_init = 970     # Initial Weighted Average Material Cost
n=59       # No. of evaluation days
#n_weeks = n/7
#n_months = round(n/30)
consp = np.ones(n)*400  # Consumption - From Production schedule by customer
consp[3] = 800
wamc = np.ones(n)
wamc[0] = WAMC_init


# Transportation multiples
lorry_mult  = 900    # MT
ship_mult   = 1400   # MT


# Inventory bounds
Inv_min = 1500
Inv_max = 9000
OI_mult = 100  # This speeds up the calculation


# Lead times
ldtimeA = 0
ldtimeB = 10
ldtimeC = 10 # Spot


# Dates
basedate = datetime.datetime.today()
# basedate = datetime.datetime(2018, 4, 2)
date_list = [basedate + datetime.timedelta(days=x) for x in range(n)]
date_list_conv = [date_list[i].strftime('%d-%b-%y') for i in range(len(date_list))]
# default format : 2019-10-22 08:11:07.561376 - ISO 8601 format (YYYY-MM-DDTHH:MM:SS.mmmmmm)


# Volume to purchase from each supplier - to be integrated with Volume Apportionment
total_volume = 30000
VA_A = round(1*total_volume)
VA_B = round(0*total_volume)
VA_C = round(0*total_volume)


# Weekly Prices - to be integrated with
price_A = [895,895,895,895,856,856,856,856]
price_B = [748,700,750,710,725,746,774,770]
price_C = [686,686,627,627,631,631,668,670]
# date = ['7-Apr-19','14-Apr-19','21-Apr-19','28-Apr-19','5-May-19','12-May-19','19-May-19']
# date_conv = [datetime.datetime.strptime(date[i],'%d-%b-%y') for i in range(len(date))]


df = pd.DataFrame({'Price A':price_A,'Price B':price_B,'Price C': price_C})
#price_min = df.min()
# df = pd.DataFrame({'Date':Date ,'Price A':price_A,'Price B':price_B,'Price C': price_C})
pa = np.ones(n)
pb = np.ones(n)
for i in range(0,n):
    if i < 29:
    # print(price_A[0])
        pa[i] = price_A[0]
        pb[i] = price_B[0]
    elif i < 59:
    # print(price_A[1])
        pa[i] = price_A[4]
        pb[i] = price_B[4]
#    elif i < 3*n/n_weeks:
#    # print(price_A[2])
#        pa[i] = price_A[2]
#        pb[i] = price_B[2]
#    elif i < 4*n/n_weeks:
#    # print(i)
#        pa[i] = price_A[3]
#        pb[i] = price_B[3]






# Previous PO
prev_PO = np.zeros(n)
#prev_PO[1] = 100
#prev_PO[6] = 60
#k = 0
"""""""""""""""""""""""
Objective Function
"""""""""""""""""""""""
def objective_function(params):
    # Initialisation
    x=np.zeros(n)   
    OI=np.ones(n)
    OI[0] = OI_init/OI_mult
        
    x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],\
    x[14],x[15],x[16],x[17],x[18],x[19],x[20],x[21],x[22],x[23],x[24],x[25],x[26],x[27],\
    x[28],x[29],x[30],x[31],x[32],x[33],x[34],x[35],x[36],x[37],x[38],x[39],x[40],x[41],\
    x[42],x[43],x[44],x[45],x[46],x[47],x[48],x[49],x[50],x[51],x[52],x[53],x[54],x[55],x[56],x[57],x[58],\
    OI[0],OI[1],OI[2],OI[3],OI[4],OI[5],OI[6],OI[7],OI[8],OI[9],OI[10],OI[11],OI[12],OI[13],\
    OI[14],OI[15],OI[16],OI[17],OI[18],OI[19],OI[20],OI[21],OI[22],OI[23],OI[24],OI[25],OI[26],OI[27],\
    OI[28],OI[29],OI[30],OI[31],OI[32],OI[33],OI[34],OI[35],OI[36],OI[37],OI[38],OI[39],OI[40],OI[41],\
    OI[42],OI[43],OI[44],OI[45],OI[46],OI[47],OI[48],OI[49],OI[50],OI[51],OI[52],OI[53],OI[54],OI[55],OI[56],OI[57],OI[58]\
    = params    # <-- for readability you may wish to assign names to the component variables
                # params is a NumPy array
    
    for i in range(1,n):
        
        OI[i] = (OI[i-1]*OI_mult + x[i-1]*lorry_mult - consp[i-1])/OI_mult
        
        wamc[i] = (OI[i]*OI_mult*wamc[i-1] + pa[i]*x[i]*lorry_mult ) \
            /(OI[i]*OI_mult + x[i]*lorry_mult)
        #CI[i] = OI[i] + x[i]*lorry_mult + y[i]*ship_mult + z[i]*ship_mult - consp[i]
#        print(wamc)
    
    def eqn1():
        
        tot1 = A*((2*OI[0]*OI_mult + x[0]*lorry_mult + prev_PO[0] - consp[0])*wamc[0])\
            +  sum ([A*(2*OI[i]*OI_mult + x[i]*lorry_mult - consp[i]) *wamc[i] for i in range (1,n)])
                   
        return tot1
    
    
    def eqn2():
        
        tot2 = pa[0]*x[0]*lorry_mult + sum ([pa[i]*x[i]*lorry_mult for i in range (1,n)])
        
        return tot2
    
    tot1 = eqn1()
    tot2 = eqn2()
    total = tot1 + tot2
    
    return total, print (f'Inventory Cost: {tot1} + '), print (f'{tot2}'),print (f'{total}\n') #,print(OI*OI_mult)


bnd_a = (0,1600/lorry_mult)  # bounds x - sup A
#bnd_b = (0,3)   # bounds y - sup B
#bnd_c = (0,4)   # bounds z - sup C
bnd_zero = (0,0)    # Lead time bounds set to 0
bnd_OI_init = (OI_init/OI_mult,OI_init/OI_mult) # bounds for OI[0]
bnd_OI = (Inv_min/OI_mult,Inv_max/OI_mult) # bounds OI
e = (bnd_zero*ldtimeA + bnd_a*(n-ldtimeA) +  bnd_OI_init + bnd_OI*(n-1))
g = [(e[i], e[i+1]) for i in range(0,len(e),2)]
bnds = tuple(g)


# inequality definition in scipy --> x[0]+x[1]+x[2]+x[3]>=1
# equality constraint for cobyla workaround - see https://stackoverflow.com/questions/35631192/element-wise-constraints-in-scipy-optimize-minimize/35631777#35631777
cons = [\
        {'type': 'ineq',\
         'fun': lambda x:  -((sum(x[0:59]))*lorry_mult - total_volume)},\
         
        {'type': 'ineq',\
         'fun': lambda x:  (sum(x[0:59]))*lorry_mult - total_volume},\
# Sup A
        {'type': 'ineq',\
         'fun': lambda x:  -((sum(x[0:29]))*lorry_mult - 12000)},\
         
        {'type': 'ineq',\
         'fun': lambda x:  (sum(x[0:29]))*lorry_mult - 7000},\
# Sup B                                             
        {'type': 'ineq',\
         'fun': lambda x:  -((sum(x[29:59]))*lorry_mult - 18000)},\
            
        {'type': 'ineq',\
         'fun': lambda x:  (sum(x[29:59]))*lorry_mult - 7000}]#,\
## Sup C
#        {'type': 'ineq',\
#         'fun': lambda z:  -((sum(z[56:84]))*ship_mult - VA_C)},\
#            
#        {'type': 'ineq',\
#         'fun': lambda z:  (sum(z[56:84]))*ship_mult - VA_C}]
                  
"""""""""""""""""""""""""""""""""""
Weekly Constraints - Sup A & B
"""""""""""""""""""""""""""""""""""
#for i in range(0,22,7):
#    # Sup A
#    # x(i) > 340
#    w1a = {'type': 'ineq',\
#         'fun': lambda x,i=i:  (x[i] + x[i+1] + x[i+2]+ x[i+3] + x[i+4] + \
#         x[i+5] + x[i+6])*lorry_mult - 350}
#    # x(i) < 480
#    w2a = {'type': 'ineq',\
#         'fun': lambda x,i=i:  -((x[i] + x[i+1] + x[i+2]+ x[i+3] + x[i+4] + \
#         x[i+5] + x[i+6])*lorry_mult - 520)}
#    
#    # Sup B
#    # x(i) > 0
#    w1b = {'type': 'ineq',\
#         'fun': lambda x,i=i:  (x[i+28] + x[i+29] + x[i+30]+ x[i+31] + x[i+32] + \
#         x[i+33] + x[i+34])*ship_mult}
#    # x(i) < 300
#    w2b = {'type': 'ineq',\
#         'fun': lambda x,i=i:  -((x[i+28] + x[i+29] + x[i+30]+ x[i+31] + x[i+32] + \
#         x[i+33] + x[i+34])*ship_mult - 300)}
#    
#    cons.append(w1a)
#    cons.append(w2a)
#    cons.append(w1b)
#    cons.append(w2b)


"""""""""""""""""""""""""""""""""""""""""""""""""""
Spot Sup C constraint - Depends on ship schedule
(integrate with shipping calendar in the future)
"""""""""""""""""""""""""""""""""""""""""""""""""""
# Spot supplier delivery dates
#ssdd = [ '15-Nov-19','30-Nov-19']
#j = [date_list_conv.index(ssdd[i]) + n*2 + 1 for i in range(len(ssdd))]
#
#for j in j:
#    w1cu = {'type': 'ineq',\
#             'fun': lambda x, j = j: x[j]*ship_mult - VA_C/2}
#    w1cl = {'type': 'ineq',\
#             'fun': lambda x, j = j:  -(x[j]*ship_mult - VA_C/2)}
#              
#    cons.append(w1cu)
#    cons.append(w1cl)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Setting Decision Variables Boundaries as Inequalities
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
for factor in range(len(bnds)):
    # print(factor)
    lower, upper = bnds[factor]
    l = {'type': 'ineq',
         'fun': lambda x, lb=lower, i=factor: x[i] - lb}
    u = {'type': 'ineq',
         'fun': lambda x, ub=upper, i=factor: ub - x[i]}
    
    cons.append(l)
    cons.append(u)


"""""""""""""""""""""""""""""""""""""""""""""
Opening Inventory(i) == Closing Inventory(i-1) -
Add this because OI and CI will go below minimum inventory if without
"""""""""""""""""""""""""""""""""""""""""""""
#consumption = np.ones(n)*100
for factor in range(1,n):
    consumption = consp[factor-1]
    k = {'type': 'ineq', 'fun': lambda x, i =factor,consumption=consumption: -(x[i+59]*OI_mult - (x[i+59-1]*OI_mult + x[i-1]*lorry_mult - consumption))}
    m = {'type': 'ineq', 'fun': lambda x, i =factor,consumption=consumption: x[i+59]*OI_mult - (x[i+59-1]*OI_mult + x[i-1]*lorry_mult - consumption)}
    cons.append(k)
    cons.append(m)


"""""""""""""""
OPTIMISATION
"""""""""""""""
initial_guess = np.concatenate((np.ones(n)*3,np.ones(n)*4))
result = optimize.minimize(objective_function, initial_guess,method='COBYLA',constraints=cons,options={'rhobeg': 1.1, 'maxiter': 50000, 'disp': False, 'catol': 0.02})
#result = optimize.minimize(f, initial_guess,method='SLSQP',constraints=cons,options={'disp': True})


if result.success:
    fitted_params = np.round(result.x,2)
    print(fitted_params)
else:
    raise ValueError(result.message)


"""""""""""
Validation
"""""""""""
# monthly
#sum(result.x[0:n])     # Sup A check
#sum(result.x[n:2*n])    # Sup B check
#sum(result.x[56:84])    # Sup C check
#sum(result.x)


# weekly - Sup A
sum(result.x[0:29])*lorry_mult
sum(result.x[29:59])*lorry_mult
#sum(result.x[14:21])*lorry_mult
#sum(result.x[21:28])*lorry_mult
sum(result.x[0:59])*lorry_mult     # Sup A check


# weekly - Sup B
#sum(result.x[28:35])*ship_mult
#sum(result.x[35:42])*ship_mult
#sum(result.x[42:49])*ship_mult
#sum(result.x[49:56])*ship_mult    # Sup B check
#sum(result.x[28:56])*ship_mult




sup_A_order = result.x[0:59]*lorry_mult
#sup_B_order = result.x[28:56]*ship_mult
#sup_C_order = result.x[56:84]*ship_mult
sum(sup_A_order)
#sum(sup_B_order)
#sum(sup_C_order)
Arnd = np.round(sup_A_order/10)*10
#Brnd = np.round(sup_B_order/10)*10
#Crnd = np.round(sup_C_order/10)*10
sum(np.round(sup_A_order/10)*10)
#sum(np.round(sup_B_order/10)*10)
#sum(np.round(sup_C_order/10)*10)
#round_replenish = sum([Arnd, Brnd, Crnd])
round_replenish = sum([Arnd])


total = sum(round_replenish)


OI_pred = result.x[59:118]*OI_mult
replenish = sup_A_order
#replenish = sup_A_order + sup_B_order + sup_C_order




CI = np.zeros(n)
CI[0] = OI_init + sup_A_order[0] - consp[0]
for i in range(1,n):
    #OI[i] = CI[i-1]
    CI[i] = int(result.x[i+59]*OI_mult + result.x[i]*lorry_mult - consp[i])
    if CI[i] > Inv_max:
        res = win32api.MessageBox(None,f"Your total inventory on day {i} is {int(CI[i])} MT, \
which is above the maximum allowable inventory level of {Inv_max} MT.\
Would you like to reduce the purchase volume?", "Subtle Warning",1)
        if res == 1:
            print ('Ok. Proceed to reduce Volume.')
        elif res == 2:
            print ('cancel. Ignore and continue with current purchased Volume')
        
OI=np.ones(n)
OI[0] = OI_init
for i in range(1,n):
    OI[i] = (OI[i-1] + result.x[i-1]*lorry_mult - consp[i-1])
#    wamc[i] = (OI[i]*wamc[i-1] + pa[i]*result.x[i]*lorry_mult + pb[i]*result.x[i+28]*ship_mult + price_min[2]*result.x[i+56]*ship_mult)/(OI[i] + result.x[i]*lorry_mult + result.x[i+28]*ship_mult + result.x[i+56]*ship_mult)


"""""""""
Plotting
"""""""""
fig1 = plt.figure(1)
fig1.set_figwidth(14)
fig1.set_figheight(7)
#ax1 = fig.add_subplot(2, 1, 1)
ax1 = plt.axes()
#ax1 = plt.subplots()
ax1.set_facecolor('0.95')
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, which='minor', linewidth=0.5, linestyle=':')
ax1.minorticks_on()
ax1.plot(date_list, replenish,'yo:',label='Total Daily Replenishment',linewidth=1)
ax1.step(date_list, sup_A_order,'rs--',label='Sup A - Domestic', linewidth = 0.5 ,markersize=2)
ax1.step(date_list, consp,'bo:',where='mid',label='Consumption',markersize=3)
ax1.text(date_list[0]-datetime.timedelta(days=0.5), consp[0], 'Consumption', verticalalignment='center', horizontalalignment='right')
formatter = DateFormatter('%a,%b-%d-%y')
ax1.xaxis.set_major_formatter(formatter)
ax1.xaxis.set_tick_params(rotation=30, labelsize=9)
#ax1.plot(date_list, sup_B_order,'g^:',label='Sup B - International', linewidth= 0.5 ,markersize=2)
#ax1.plot(date_list, sup_C_order,'bo:',label='Sup C - Spot', linewidth = 0.5 ,markersize=2)


"""""""""
Bar Plot
"""""""""
# ref https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
width = 0.35
rects1 = ax1.bar(date_list,Arnd,width=-width,align='edge',label='Sup A')
#rects2 = ax1.bar(date_list,Brnd,width, align='edge',label='Sup B')
#rects3 = ax1.bar(date_list,Crnd,width/2,label='Sup C')
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(round(rect.get_height()))
        ax1.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
autolabel(rects1)
#autolabel(rects2)
#autolabel(rects3)


ax1.set_xlim(date_list[0]-datetime.timedelta(days=3),date_list[n-1]+datetime.timedelta(days=3))
ax1.legend()
ax1.set_title(f"Suggested Optimised Order Quantity across {n} days (Monthly contract)\n\
Month1 = {int(round(sum(result.x[0:29])*lorry_mult))} MT, Month2 = {int(round(sum(result.x[29:59])*lorry_mult,0))} MT\n\
Material cost = USD 26148000 + \nInventory Holding cost = USD 25856 \n = Total Cost: USD {int(result.fun)}",\
                  fontweight="bold")
#ax1.set_xlabel("Days of the month\n",fontweight="bold")
ax1.set_ylabel("Amount of purchased commodity in Metric Tan (MT)\n", fontweight="bold")


"""""""""""""""""""""""""""""""""
Plotting inventory
"""""""""""""""""""""""""""""""""
fig2 = plt.figure(2)
fig2.set_figwidth(14)
fig2.set_figheight(7)
#ax1 = fig.add_subplot(2, 1, 1)
ax2 = plt.axes()
#ax1 = plt.subplots()
ax2.set_facecolor('0.95')
ax2.grid(b=True, which='major', linestyle='-')
ax2.grid(b=True, which='minor', linewidth=0.5, linestyle=':')
ax2.minorticks_on()


ax2.step(date_list, OI_pred,'mo:',where='mid',label='Optimised OI',markersize=3)
ax2.step(date_list, CI,'ko-',label='Closing Inventory',where='mid',linewidth=1,markersize=3)
ax2.plot(date_list, OI,'r2:',label='Opening Inventory (Manual)',linewidth=1)
formatter = DateFormatter('%a,%b-%d-%y')
ax2.xaxis.set_major_formatter(formatter)
ax2.xaxis.set_tick_params(rotation=30, labelsize=9)


ax2.plot([date_list[0], date_list[n-1]],[Inv_min,Inv_min],'r:')
ax2.text(date_list[0],Inv_min , 'Minimum Inventory Level', verticalalignment='top', horizontalalignment='left')


ax2.plot([date_list[0], date_list[n-1]],[Inv_max,Inv_max],'r:')
ax2.text(date_list[0], Inv_max, 'Maximum Inventory Level', verticalalignment='bottom', horizontalalignment='left')


ax2.text(date_list[0]-datetime.timedelta(days=0.5), CI[0], 'Closing\nInventory', verticalalignment='center', horizontalalignment='right')


ax2.text(date_list[0]-datetime.timedelta(days=0.5), OI[0], 'Opening\nInventory', verticalalignment='center', horizontalalignment='right')


ax2.set_xlim(date_list[0]-datetime.timedelta(days=3),date_list[n-1]+datetime.timedelta(days=3))
ax2.legend()
ax2.set_title(f"Opening and Closing Inventory in {n} days (Monthly contract)\n\
\nTotal Cost: USD {int(result.fun)}",\
                  fontweight="bold")
#ax1.set_xlabel("Days of the month\n",fontweight="bold")
ax2.set_ylabel("Amount of commodity in Inventory (MT)\n", fontweight="bold")