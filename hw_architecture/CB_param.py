import numpy as np
import torch
import pandas as pd

class CB_param():
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.cb_size = (256, 64) # (rows, cols)
        
        #########################################
        ## RRAM for inference
        self.high_precision = False
        if self.high_precision:
            self.weight_bits = 32
        else:
            self.weight_bits = 4  # programming target resolution
        self.cell_I_max = 3e-6
        self.cell_I_min = 0.3e-6
        self.cell_I_short = 30e-6 ## not used
        self.short_ratio = 0.002 ## not used
        
        self.onoff_ratio_ideal = False
        self.onoff_ratio = 10 #(infinite, 10, 100, 1000) for training
        
        self.if_weight_noise = True 
        self.weight_noise_type = "sigma_relative"
        self.weight_noise_sigma = 2 
        self.weight_noise_rel = 1/(2**self.weight_bits - 1)*self.weight_noise_sigma # weight variation relative to dynamic range (I_max - I_min)
        
        ################################################################################
        ## ADC Bit-serial
        self.inp_bits = 8
        self.ADC_bits = 8 
        if self.high_precision:
            self.ADC_bits = 32 # for digital training and inference
     
        self.ADC_range = 45e-6 # MAX accumulated current if analog innput of [0, 1] is used. 
        self.ADC_usable_range = 30e-6 # ADC_range subtract the offset current, the actual usable range
        self.if_ADC_noise = True
        self.ADC_noise_type = "sigma_relative"
        self.ADC_noise_sigma = 2 # std between ADC levels for added noise
        self.ADC_noise_rel = 1/(2**self.ADC_bits-1)*self.ADC_noise_sigma # ADC front-end noise level 2 sigma spacing
        self.ADC_noise = self.ADC_noise_rel * self.ADC_range
        ################################################################################
        ## weight range: [1/onoff_ratio, 1.0]
        ## ADC range: [0, ADC_range/w_max]
        self.weight_min_train = 1/self.onoff_ratio
        self.weight_max_train = 1.0
        self.weight_range_train = self.weight_max_train - self.weight_min_train
        self.weight_noise = self.weight_noise_rel * self.weight_range_train
        self.ADC_range_train = self.ADC_range/self.cell_I_max
        self.ADC_usable_range_train = self.ADC_usable_range/self.cell_I_max
        self.ADC_noise_train = self.ADC_noise/self.cell_I_max
        
        ## Mixed precision training
        self.deltaw_threshold = self.weight_range_train/(2**self.weight_bits-1)
        
        self.if_weight_prog_noise = True
        self.weight_prog_bits = self.weight_bits # weight prog bits for mixed precision training
        self.weight_prog_noise_sigma = 2 # std between levels for added noise 
        self.weight_prog_noise_rel = 1/(2**self.weight_prog_bits - 1)*self.weight_prog_noise_sigma
        self.weight_prog_noise = self.weight_prog_noise_rel * self.weight_range_train
        
        ################################################################################
        self.update()
        
    def update(self):
        ################################################################################
        # RRAM Cell for inference
        ################################################################################
        if self.high_precision:
            self.weight_bits = 32 # tf fake_quant function support up to 16 bits, use 15+1 bits for full range symc quant
            
        self.cell_I_min = self.cell_I_max/self.onoff_ratio
        self.weight_noise_rel = 1/(2**self.weight_bits - 1)*self.weight_noise_sigma
        
        ################################################################################
        # ADC
        ################################################################################
        if self.high_precision:
            self.ADC_bits = 32 # for digital training and inference
    
        self.ADC_noise_rel = 1/(2**self.ADC_bits-1)*self.ADC_noise_sigma # ADC front-end noise level 2 sigma spacing
        self.ADC_noise = self.ADC_noise_rel * self.ADC_range
        self.ADC_range_train = self.ADC_range/self.cell_I_max
        self.ADC_usable_range_train = self.ADC_usable_range/self.cell_I_max
        self.ADC_noise_train = self.ADC_noise/self.cell_I_max
        
        ################################################################################
        # Training 
        ################################################################################
        self.weight_min_train = 1/self.onoff_ratio
        self.weight_max_train = 1.0
        self.weight_range_train = self.weight_max_train - self.weight_min_train
        self.weight_noise = self.weight_noise_rel * self.weight_range_train
        self.deltaw_threshold = self.weight_range_train/(2**self.weight_bits-1)
        
        self.weight_prog_noise_rel = 1/(2**self.weight_prog_bits - 1)*self.weight_prog_noise_sigma
        self.weight_prog_noise = self.weight_prog_noise_rel * self.weight_range_train
        
    def summary(self):
        ## Print summary
        print("=========================================")
        print("Array size:\t\t", self.cb_size)
        print("Input bits:\t\t", self.inp_bits)
        print("Weight bits:\t\t", self.weight_bits)
        print("Act bits:\t\t", self.ADC_bits)
        print("-----------------------------------------")
        print("onoff_ratio:\t\t", self.onoff_ratio)
        print("weight_train_range:\t(%.2f-%.2f)" %(self.weight_min_train, self.weight_max_train)) 
        print("if_weight_noise:\t", self.if_weight_noise)
        print("weight_noise:\t\t", self.weight_noise)
        print("weight_prog_bits:\t\t", self.weight_prog_bits)
        print("weight_prog_noise:\t\t", self.weight_prog_noise)
        print("-----------------------------------------")
        print("ADC_range_train:\t\t", self.ADC_range_train)
        print("ADC_usable_range_train:\t\t", self.ADC_usable_range_train)
        print("if_ADC_noise:\t\t\t", self.if_ADC_noise)
        print("ADC_noise_train:\t\t", self.ADC_noise_train)
        print("=========================================")

    def save_to_file(self, file_path = '../saved_models/', file_name = None):
        ## store parameters to excel file
        attrs = vars(self)
        Dict = {}
        for key, val in attrs.items():
            Dict[key] = [val]
        df = pd.DataFrame(Dict)
        if file_name is None:
            file_name = "CB_param"
        df.to_csv(file_path + file_name + ".csv", index=False)

gp = CB_param()        