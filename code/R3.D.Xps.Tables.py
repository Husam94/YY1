''' THIS SCRIPT IS TO SAVE THE XPS TABLES AS FIGURES FOR PLOS ONE '''

#===============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import glob, shutil, os

#------------------------------------

from torch.optim import Adam, AdamW

#------------------------------------

import poseigen_seaside.basics as se
import poseigen_seaside.metrics as mex

import poseigen_compass as co

import poseigen_binmeths as bm

import poseigen_trident.utils as tu

import poseigen_oyster as oys

#------------------------------------

import R3_xps_functions as xpfus

#------------------------------------

import dataframe_image as dfi
from PIL import Image, ImageDraw, ImageFont


#=====================================================

data_path = "../data/R3/"
os.makedirs(data_path, exist_ok=True)

os.chdir(data_path)



fin_bins = se.PickleLoad('fin_bins')

divset = ['Train', 'Stop.', ' Eval', 'Test']

xpsfolder = se.NewFolder('xps_4')
tables_dir = se.NewFolder(xpsfolder + 'tables2')


#**********************************************

RO_folder = se.NewFolder('RO')

ex_oys_folder = se.NewFolder(RO_folder + 'exact')
oys_iter = 11
pn_RO_exact = se.NewFolder(ex_oys_folder + 'RO_' + str(oys_iter))
pn_RO_exact_top = se.NewFolder(pn_RO_exact + 'Top' + str(0))
exact_ranked_can_dict = se.PickleLoad(pn_RO_exact + 'ranked_can_dict')

exact_dict = exact_ranked_can_dict[0]

#---------------------------------------------

nonex_oys_folder = se.NewFolder(RO_folder + 'nonexact')
ne_iter = 11
pn_RO_ne = se.NewFolder(nonex_oys_folder + 'RO_' + str(ne_iter))
pn_RO_ne_top = se.NewFolder(pn_RO_ne + 'Top' + str(0))
ne_ranked_can_dict = se.PickleLoad(pn_RO_ne + 'ranked_can_dict')

ne_dict = ne_ranked_can_dict[0]

#---------------------------------------------

allconfigs = {'Exact': exact_dict, 'non-Exact': ne_dict}



yy1_split_transaug = se.PickleLoad('yy1_split_transaug')

yy1_max = se.PickleLoad('yy1_max')

tri_tpacks = se.PickleLoad('tri_tpacks')

d_x1, d_y_ms, d_s, d_b, d_x2 = tri_tpacks

d_s_rs = np.swapaxes(np.expand_dims(d_s, axis = -1), 1, -1)



masterpseudo = 1e-10

RMS_mode = [mex.MeanExpo, {'expo': 2, 'root': True}]
RMS_mode_pyt = [mex.MeanExpo, {'expo': 2, 'root': True, 'pyt': True}]

bm_args_np = {'byaxis': 1, 'useweights': False, 'seperate': False, 
              'summarize_mode': RMS_mode}

#------------------------------------------------

deverr_args_base = {'expo': 1, 'root': False,                                                       #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
                    'pseudo': masterpseudo,
                    'scalefactor': yy1_max}

MDE_mode_np = [mex.DeviaError, {**deverr_args_base, 'pyt': False}]

B_MDE_mode_np = [tu.BinnedLoss, {'metrics_mode': MDE_mode_np, **bm_args_np}]

TCR_repeats = 15


#--------------------------------------------------------------------------

es_args = {'out': d_y_ms, 'out_std': d_s_rs, 'out_bind': d_b,
            'split': yy1_split_transaug,
            'metrics_mode': B_MDE_mode_np, 
            'score_on': 1,
            'std_cutoff': None, 'ddof': 1, 'top': 3, 'smallest': True}




xp1_trial = 3                   # FROM 2 
xp3_trial = 3                   # FROM 2
xp5_trial = 3                   # FROM 2
xp6_trial = 3                   # FROM 2


xp1_id, xp3_id, xp5_id, xp6_id = ['xp' + str(x) + '_' + str(y) 
                                  for x,y in zip([1, 3, 5, 6], 
                                                 [xp1_trial, xp3_trial, xp5_trial, 
                                                  xp6_trial])]

pn_xp1, pn_xp3, pn_xp5, pn_xp6 = [se.NewFolder(xpsfolder + ido) 
                                            for ido in [xp1_id, xp3_id, xp5_id, xp6_id]]

#===========================================================

def style_all_borders(styler):
    return styler.set_table_styles(
        [
            {'selector': 'th, td', 'props': [
                ('border', '1px solid black'),
                ('font-family', '"Times New Roman", Times, serif'),
                ('font-size', '11pt'),
                ('text-align', 'center'),
                ('vertical-align', 'middle')
            ]},
            {'selector': 'th', 'props': [('font-weight', 'bold')]}
        ],
        overwrite=False
    ).set_properties(**{
        'text-align': 'center',
        'vertical-align': 'middle',
        'font-family': '"Times New Roman", Times, serif',
        'font-size': '11pt'
    })


#******************************************************************************************

pn_xp1 = se.NewFolder(xpsfolder + 'xp1' + '_' + str(xp1_trial))

xp1_variables = ['Loss Function']

lossos = ['MDE', 'RBM', 'Shuffled MDE']

xp1_combs = [[lo] for lo in lossos]

xp1_combs = [[ic, c] for ic,c in enumerate(xp1_combs)]


xp1_bs_all, xp1_bs_r2r_all = xpfus.XpsResults(pn_xp1, xp1_combs, 
                                              TCR_repeats, es_args, allconfigs,
                                              
                                              ref_ic = 0, rewrite = False)

xp1_tabs = xpfus.XpsTables(pn_xp1, xp1_combs, xp1_variables, xp1_bs_r2r_all, allconfigs)

xp1_tab_main, xp1_tab_main_sty, xp1_tab_extra, xp1_tab_extra_sty = xp1_tabs

xp1_tab_main_sty = style_all_borders(xp1_tab_main_sty)
xp1_tab_extra_sty = style_all_borders(xp1_tab_extra_sty)

for x, y in zip([xp1_tab_main_sty, xp1_tab_extra_sty], ['main', 'extra']):
    
    fig_loc = f'{tables_dir}/xp1_{y}.tif'

    dfi.export(x, fig_loc, dpi=300)

    img = Image.open(fig_loc)
    scale = 0.75
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, resample=Image.LANCZOS)
    img_resized.save(fig_loc, format='TIFF', dpi=(300, 300))

#******************************************************************************************

pn_xp3 = se.NewFolder(xpsfolder + 'xp3' + '_' + str(xp3_trial))

xp3_variables = ['Weighting']

binweis = ['None', 'DenseW', 'Recip.']
xp3_combs = []

for bw in binweis: xp3_combs.append([bw])

xp3_combs = [[ic, c] for ic,c in enumerate(xp3_combs)]

xp3_bs_all, xp3_bs_r2r_all = xpfus.XpsResults(pn_xp3, xp3_combs, 
                                              TCR_repeats, es_args, allconfigs,
                                              
                                              ref_ic = 0, rewrite = False)

xp3_tabs = xpfus.XpsTables(pn_xp3, xp3_combs, xp3_variables, xp3_bs_r2r_all, allconfigs)

xp3_tab_main, xp3_tab_main_sty, xp3_tab_extra, xp3_tab_extra_sty = xp3_tabs

xp3_tab_main_sty = style_all_borders(xp3_tab_main_sty)
xp3_tab_extra_sty = style_all_borders(xp3_tab_extra_sty)

for x, y in zip([xp3_tab_main_sty, xp3_tab_extra_sty], ['main', 'extra']):
    
    fig_loc = f'{tables_dir}/xp3_{y}.tif'

    dfi.export(x, fig_loc, dpi=300)

    img = Image.open(fig_loc)
    scale = 0.75
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, resample=Image.LANCZOS)
    img_resized.save(fig_loc, format='TIFF', dpi=(300, 300))


#******************************************************************************************

us_props = ['None'] + [str(x) for x in [0.8, 0.4, 0.2, 0.1, 0.05, 0.025, 0.01]]

pn_xp5 = se.NewFolder(xpsfolder + xp5_id)

xp5_variables = ['Samp. Prop.'] 

xp5_combs = []
for us in us_props: xp5_combs.append([us])
    
xp5_combs = [[ic, c] for ic,c in enumerate(xp5_combs)]

xp5_bs_all, xp5_bs_r2r_all = xpfus.XpsResults(pn_xp5, xp5_combs, 
                                              TCR_repeats, es_args, allconfigs,
                                              
                                              ref_ic = 0, rewrite = False)

xp5_tabs = xpfus.XpsTables(pn_xp5, xp5_combs, xp5_variables, xp5_bs_r2r_all, allconfigs)

xp5_tab_main, xp5_tab_main_sty, xp5_tab_extra, xp5_tab_extra_sty = xp5_tabs

xp5_tab_main_sty = style_all_borders(xp5_tab_main_sty)
xp5_tab_extra_sty = style_all_borders(xp5_tab_extra_sty)

for x, y in zip([xp5_tab_main_sty, xp5_tab_extra_sty], ['main', 'extra']):
    
    fig_loc = f'{tables_dir}/xp5_{y}.tif'

    dfi.export(x, fig_loc, dpi=300)

    img = Image.open(fig_loc)
    scale = 0.75
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, resample=Image.LANCZOS)
    img_resized.save(fig_loc, format='TIFF', dpi=(300, 300))

#******************************************************************************************

pn_xp6 = se.NewFolder(xpsfolder + xp6_id)

xp6_variables = ['Loss Function', 'Weighting', 'Samp. Prop.']

top_lf = ['MDE', 'RBM']
top_weis = ['None', 'Recip.']
top_us = [False, True]

xp6_combs = []

for los in top_lf: 

    for bw in top_weis: 
        
        for us in top_us:
        
            xp6_combs.append([los, bw, us])

xp6_combs = [[ic, c] for ic,c in enumerate(xp6_combs)]

xp6_bs_all, xp6_bs_r2r_all = xpfus.XpsResults(pn_xp6, xp6_combs, 
                                              TCR_repeats, es_args, allconfigs,
                                              
                                              ref_ic = 0, rewrite = False)

xp6_tabs = xpfus.XpsTables(pn_xp6, xp6_combs, xp6_variables, xp6_bs_r2r_all, allconfigs)

xp6_tab_main, xp6_tab_main_sty, xp6_tab_extra, xp6_tab_extra_sty = xp6_tabs

xp6_tab_main_sty = style_all_borders(xp6_tab_main_sty)
xp6_tab_extra_sty = style_all_borders(xp6_tab_extra_sty)

for x, y in zip([xp6_tab_main_sty, xp6_tab_extra_sty], ['main', 'extra']):
    
    fig_loc = f'{tables_dir}/xp6_{y}.tif'

    dfi.export(x, fig_loc, dpi=300)

    img = Image.open(fig_loc)
    scale = 0.75
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, resample=Image.LANCZOS)
    img_resized.save(fig_loc, format='TIFF', dpi=(300, 300))