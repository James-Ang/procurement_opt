# -*- coding: utf-8 -*-
"""
Created on Wed May  6 10:46:57 2020

@author: User
"""

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
# import sys
import time
from pro_opt_lib import proc_opt

app = Flask(__name__)
api = Api(app)


class index(Resource):
    def get(self):
        return request.url
    
class test(Resource):
    def post(self):
        a = request.form['br']
        sup1 = request.form['sup1_tier_weight']
        print(type(int(a)))
        sup1_int = [int(x) for x in sup1.split(',')]
        # print(type(int(sup1_int)))
        return jsonify({"rw":sup1_int})

class procure(Resource):
    
    def post(self):
               
        # Start Time
        start_time = time.time()
        
        inv_start_point = int(request.form['inv_start_point'])
        
        ini_weig_mat_cost = int(request.form['ini_weig_mat_cost'])
        
        num_eval_days = int(request.form['num_eval_days'])
        
        inv_min = int(request.form['inv_min'])
        
        inv_max = int(request.form['inv_max'])
        
        leadtimeA = int(request.form['leadtimeA'])
        
        total_volume = int(request.form['total_volume'])
        
        perc_purvol_A = int(request.form['percent_purchase_vol_A'])
        
        perc_purvol_B = int(request.form['percent_purchase_vol_B'])
        
        prA = request.form['weekly_price_A']
        price_A = [int(x) for x in prA.split(',')]
        
        prB = request.form['weekly_price_B']
        price_B = [int(x) for x in prB.split(',')]
        
        prC = request.form['weekly_price_C']
        price_C = [int(x) for x in prC.split(',')]
        
        # sup2_weight = request.form['sup2_tier_weight']
        # sup2_tier_weight = [int(x) for x in sup2_weight.split(',')]
        
        # sup3_weight = request.form['sup3_tier_weight']
        # sup3_tier_weight = [int(x) for x in sup3_weight.split(',')]
        
        # sup1_tier = request.form['sup1_tier_price']
        # sup1_tier_price = [int(x) for x in sup1_tier.split(',')]
        
        # sup2_tier = request.form['sup2_tier_price']
        # sup2_tier_price = [int(x) for x in sup2_tier.split(',')]
        
        # sup3_tier = request.form['sup3_tier_price']
        # sup3_tier_price = [int(x) for x in sup3_tier.split(',')]
        
        # sup1_vol_appor = request.form['sup1_VA']
        # sup1_VA = [float(x) for x in sup1_vol_appor.split(',')]
        
        # sup2_vol_appor = request.form['sup2_VA']
        # sup2_VA = [float(x) for x in sup2_vol_appor.split(',')]
        
        retJSON = proc_opt(
            start_time,
            inv_start_point,
            ini_weig_mat_cost,
            num_eval_days,
            inv_min,
            inv_max,
            leadtimeA,
            total_volume,
            perc_purvol_A,
            perc_purvol_B,
            price_A,
            price_B,
            price_C,
            )
                     
        return jsonify(retJSON)

api.add_resource(index, '/')
api.add_resource(test, '/test')
api.add_resource(procure, '/procure_opt')

if __name__ == '__main__':
    app.run(debug=True)