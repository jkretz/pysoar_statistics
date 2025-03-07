import pandas as pd
import os
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy.ma as ma
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import sqlite3


def main():

    ipath_data = '/home/jkretzs/personal/pysoar/PySoar/PySoar/bin/'
    data = PysoarXlsToDict(ipath_data).data_read
    data_days = data['pribina-cup-2015']['15-meter']

    con = sqlite3.connect('../pysoar_statistic_old/pysoar.db')
    for day in data_days.keys():
        data = pd.DataFrame(data_days[day])
        data.to_sql(name='data', con=con, if_exists='append', index=False)
    con.close()


class PysoarXlsToDict:

    def __init__(self, ipath):
        self.ipath = ipath
        self.data_read = self.pysoar_to_dict()

    def pysoar_to_dict(self):
        data_dict = {}

        # Loop trough all available competitions
        for dirpath, _, filenames in os.walk(self.ipath):
            for filename in [f for f in filenames if f.endswith('.xls')]:

                # Get information on the available competitions
                file_path = os.path.join(dirpath, filename)
                comp_date, comp_year, comp_class, comp_name = self.competition_info(file_path)

                # Setting up the general structure of the dictionary that contains all data
                if comp_name not in data_dict.keys():
                    data_dict[comp_name] = {}
                if comp_class not in data_dict[comp_name].keys():
                    data_dict[comp_name][comp_class] = {}
                if comp_date not in data_dict[comp_name][comp_class].keys():
                    data_dict[comp_name][comp_class][comp_date] = {}

                # Import data and fill dictionary
                data_dict[comp_name][comp_class][comp_date] = self.import_day_xls(file_path)

        return data_dict

    @staticmethod
    def import_day_xls(ifile):
        df = pd.read_excel(ifile, header=1, skiprows=[2, 3]).drop('Airplane', axis=1)
        return df.to_dict()

    @staticmethod
    def competition_info(ipath):
        ipath_in_split = ipath.split('/')
        date_out = ipath_in_split[-2]
        if re.compile(re.compile('.*-.*-')).match(date_out):
            year_out = date_out.split('-')[-1]
        else:
            raise ValueError('Second to last entry is not a date')
        comp_class_out = ipath_in_split[-3]
        comp_out = ipath_in_split[-4]
        return date_out, year_out, comp_class_out, comp_out


if __name__ == "__main__":
    main()

#
#
# class SelectComps:
#
#     def __init__(self, data_in, **kwargs):
#
#         self.selection = list()
#         # Do not select specific competitor
#         for year in data_in.keys():
#             for comp in data_in[year]:
#                 for comp_class in data_in[year][comp]:
#                     if 'class_in' not in kwargs.keys():
#                         self.selection.append([year, comp, comp_class, np.nan])
#                     else:
#                         if comp_class in kwargs['class_in']:
#                             self.selection.append([year, comp, comp_class, np.nan])
#
#         # Select specific competition and competitor via command line
#         if not self.selection:
#             self.select_year(data_in)
#             self.select_competition(data_in)
#             self.select_competitor(data_in)
#
#     @ staticmethod
#     def select_from_string(sel_list, sel_info, **kwargs):
#         check_input = True
#         while check_input:
#             if 'sel_string' in kwargs.keys():
#                 sel_str = kwargs['sel_string']
#             else:
#                 sel_str = input()
#             sel_string_split = sel_str.split(',')
#             if str(len(sel_list)) in sel_string_split:
#                 return range(len(sel_list))
#             sel_out = list()
#             for selector in sel_string_split:
#                 sel_out.append(int(selector))
#             if all(np.array(sel_out) < len(sel_list)):
#                 return sel_out
#             else:
#                 if 'sel_string' in kwargs.keys():
#                     raise ValueError('Invalid '+sel_info+' chosen')
#                 else:
#                     print('Invalid '+sel_info+' chosen')
#
#     def select_year(self, data_in):
#         print('')
#         year_list = sorted(data_in.keys())
#         print('Please select year (for multiple years, input has to be seperated by ,) :')
#         for ny, year in enumerate(year_list):
#             print(str(ny)+': '+year)
#         print(str(len(year_list))+': all')
#         year_int_list = self.select_from_string(year_list, 'year')
#         for year_int in year_int_list:
#             self.selection.append(year_list[year_int])
#
#     def select_competition(self, data_in):
#         print('')
#         selection_local = list()
#         comp_list = list()
#         nc = 0
#         print('Please select competition (for multiple competitions, input has to be seperated by ,) :')
#         for sel_year in self.selection:
#             for sel_comp in data_in[sel_year].keys():
#                 comp_list.append([sel_year, sel_comp])
#                 print(str(nc)+': '+sel_comp+' / '+sel_year)
#                 nc += 1
#         print(str(len(comp_list))+': all')
#         year_int_list = self.select_from_string(comp_list, 'year')
#         for year_int in year_int_list:
#             selection_local.append(comp_list[year_int])
#         self.selection = selection_local
#
#     def select_competitor(self, data_in):
#         print('')
#         selection_local = list()
#         for sel_comp in self.selection:
#             loop_bool = True
#             while loop_bool:
#                 cid = ''
#                 if not cid:
#                     print('Enter Competition Id for '+sel_comp[1]+' :')
#                     cid = input()
#                 for sel_class in data_in[sel_comp[0]][sel_comp[1]]:
#                     competitor_list, competitor_list_upper = list(), list()
#                     for day in data_in[sel_comp[0]][sel_comp[1]][sel_class]:
#                         for competitor in data_in[sel_comp[0]][sel_comp[1]][sel_class][day]['Callsign']:
#                             competitor_list.append(
#                                 str(data_in[sel_comp[0]][sel_comp[1]][sel_class][day]['Callsign'][competitor]))
#                             competitor_list_upper.append(
#                                 str(data_in[sel_comp[0]][sel_comp[1]][sel_class][day]['Callsign'][competitor]).upper())
#                     if cid.upper() in set(competitor_list_upper):
#                         comp_class = sel_class
#                         loop_bool = False
#                         for nid, cid_list in enumerate(set(competitor_list)):
#                             if cid_list.upper() == cid.upper():
#                                 cid_out = list(set(competitor_list))[nid]
#                 if loop_bool:
#                     print('Entered Competition ID not available')
#             selection_local.append([sel_comp[0], sel_comp[1], comp_class, cid_out])
#         self.selection = selection_local
#
#
# class SelectCompetitors:
#
#     def __init__(self, selection_in, data_in):
#
#         self.data_pilot, self.data_first, self.data_best, self.data_all = {}, {}, {}, {}
#
#         for comp_info in selection_in:
#             data_comp = data_in[comp_info[0]][comp_info[1]][comp_info[2]]
#             days = list(data_comp.keys())
#             for day in days:
#                 data_day = data_comp[day]
#                 for competitor in [comp_info[3], 'first', 'best', 'all']:
#                     if competitor == comp_info[3]:
#                         self.data_pilot[day+comp_info[2]] = self.select_competitors(data_day, competitor, comp_info[3])
#                     elif competitor == 'first':
#                         self.data_first[day+comp_info[2]] = self.select_competitors(data_day, competitor, comp_info[3])
#                     elif competitor == 'best':
#                         self.data_best[day+comp_info[2]] = self.select_competitors(data_day, competitor, comp_info[3])
#                     elif competitor == 'all':
#                         comp_all = {}
#                         for comp_id in data_day['Callsign'].values():
#                             comp_all[comp_id] = self.select_competitors(data_day, comp_id, comp_id)
#                         self.data_all[day+comp_info[2]] = comp_all
#
#     def select_competitors(self, data_day_in, competitor_in, comp_num_in):
#         data_out = {}
#         unused_keys = ['Ranking', 'Callsign', 'Start time', 'Finish time',
#                        'Height loss during circling', 'Average thermal speed (GS)']
#         if competitor_in == comp_num_in:
#             for rank_callsign in data_day_in['Callsign'].items():
#                 if competitor_in in rank_callsign:
#                     day_rank = rank_callsign[0]
#                     for key in data_day_in.keys():
#                         if key not in unused_keys:
#                             data_out[key] = self.write_competitor_average(data_day_in, key, rank=day_rank, avg=False)
#                     return data_out
#         elif competitor_in == 'first':
#             day_rank = 0
#             for key in data_day_in.keys():
#                 if key not in unused_keys:
#                     data_out[key] = self.write_competitor_average(data_day_in, key, rank=day_rank, avg=False)
#             return data_out
#         elif competitor_in == 'best':
#             for key in data_day_in.keys():
#                 if key not in unused_keys:
#                     data_out[key] = self.write_competitor_average(data_day_in, key, rank=competitor_in, avg=True)
#         return data_out
#
#     @staticmethod
#     def write_competitor_average(data_in, key_in, **kwargs):
#         best_len = 3
#         rank_list = []
#         if not kwargs['avg']:
#             if key_in == 'Excess distance covered':
#                 dist_task = data_in['Distance covered from task'][kwargs['rank']]
#                 detour = 1 + (data_in['Excess distance covered'][kwargs['rank']] / 100)
#                 # return (dist_task * detour)-dist_task
#                 return data_in['Excess distance covered'][kwargs['rank']]
#             else:
#                 return data_in[key_in][kwargs['rank']]
#         elif kwargs['avg'] and kwargs['rank'] in ['all', 'best']:
#             if kwargs['rank'] == 'all':
#                 rank_list = np.arange(len(data_in['Ranking']))
#             elif kwargs['rank'] == 'best':
#                 rank_list = np.arange(best_len)
#             avg_tmp = np.empty_like(rank_list, dtype=float)
#             avg_tmp[:] = np.NaN
#             for rank in rank_list:
#                 if key_in == 'Excess distance covered':
#                     dist_task = data_in['Distance covered from task'][rank]
#                     detour = 1 + (data_in['Excess distance covered'][rank] / 100)
#                     if not np.isnan(detour) and not np.isnan(dist_task):
#                         # avg_tmp[rank] = (dist_task * detour) - dist_task
#                         avg_tmp[rank] = data_in['Excess distance covered'][rank]
#                 else:
#                     if not np.isnan(data_in[key_in][rank]):
#                         avg_tmp[rank] = data_in[key_in][rank]
#             if np.isnan(avg_tmp).all():
#                 return np.NaN
#             else:
#                 return np.nanmean(avg_tmp)
#         else:
#             raise ValueError('Can not average a single competitor')
#
#
# # This path has to point to the "bin" directory of PySoar that contains the *.xls files
# ipath_data = '/home/jkretzs/personal/pysoar/PySoar/PySoar/bin/'
# comp_class = {'std': ['STD', 'standard', 'Std.', 'Standardklasse', 'Standard'], '18m': ['18m', '18-meter'],
#               'open': ['OK', 'open', 'offene']}
#
# comp_class_use = '18m'
# data = PysoarXlsToDict(ipath_data).data_read
# selection = SelectComps(data, class_in=comp_class[comp_class_use]).selection
# select_competitors = SelectCompetitors(selection, data)
# pickle.dump(select_competitors, open('pickle/save_comps_' + comp_class_use+'.p', 'wb'))
# select_competitors = pickle.load(open('pickle/save_comps_' + comp_class_use+'.p', 'rb'))
#
#
# def normalize_pilot(data_pilot_in, data_best_in):
#     rel_best, abs_best = {}, {}
#     for key in data_pilot_in.keys():
#         abs_best[key] = data_pilot_in[key] - data_best_in[key]
#         rel_best[key] = abs_best[key] / data_best_in[key]
#     return abs_best, rel_best
#
#
# data_all_best_abs, data_all_best_rel, data_all_best = {}, {}, {}
# for var in select_competitors.data_best[list(select_competitors.data_best.keys())[0]].keys():
#     if var not in ['Start height']:
#         data_all_best_abs[var], data_all_best_rel[var], data_all_best[var] = [], [], []
#
#
# for day_all in select_competitors.data_all.keys():
#     task_speed_best = select_competitors.data_best[day_all]['Task speed']
#     for pilot in select_competitors.data_all[day_all].keys():
#         if 'Task speed' in select_competitors.data_all[day_all][pilot].keys():
#             task_speed_pilot = select_competitors.data_all[day_all][pilot]['Task speed']
#         else:
#             task_speed_pilot = np.nan
#         for var in data_all_best_abs.keys():
#             if np.abs((task_speed_pilot/task_speed_best) - 1) < 0.3:
#                 val_rel = select_competitors.data_all[day_all][pilot][var]/select_competitors.data_best[day_all][var]
#                 val_abs = select_competitors.data_all[day_all][pilot][var] - select_competitors.data_best[day_all][var]
#                 if np.abs(val_rel-1) < 1.0:
#                     data_all_best_abs[var].append(val_abs)
#                     data_all_best_rel[var].append(val_rel-1)
#                     data_all_best[var].append(select_competitors.data_best[day_all][var])
#                 else:
#                     data_all_best_abs[var].append(np.nan)
#                     data_all_best_rel[var].append(np.nan)
#                     data_all_best[var].append(np.nan)
#             else:
#                 data_all_best_abs[var].append(np.nan)
#                 data_all_best_rel[var].append(np.nan)
#                 data_all_best[var].append(np.nan)
#
# df_abs, df_rel, df_best = pd.DataFrame(data_all_best_abs), pd.DataFrame(data_all_best_rel), pd.DataFrame(data_all_best)
#
# # A filter could be placed here in the future
# df_abs, df_rel, df_best = df_abs.dropna(axis=0), df_rel.dropna(axis=0), df_best.dropna(axis=0)
#
# var_units = {'Average rate of climb': '(m/s)', 'Average cruise speed (GS)': '(km/h)', 'Average cruise distance': '(km)',
#              'Average cruise height difference': '(m)', 'Average L/D': '', 'Excess distance covered': '(%)',
#              'Task speed': '(km/h)', 'Percentage turning': '(%)'}
#
#
# val_reg = {}
# binw = 10
# speed_bins = np.arange(70, 140, binw)
# for ns, speed_bin in enumerate(speed_bins):
#     masks = [df_best['Task speed'] < speed_bin+binw/2, df_best['Task speed'] >= speed_bin-binw/2]
#     fullmask = [all(mask) for mask in zip(*masks)]
#     df_use = df_abs[fullmask]
#     y = df_use['Task speed']
#     df_use = df_use.drop(['Task speed', 'Distance covered from task'], axis=1)
#     df_use = df_use.drop('Average cruise distance', axis=1)
#     df_use = df_use.drop('Average cruise height difference', axis=1)
#     df_use = df_use.drop('Percentage turning', axis=1)
#     if df_use.size == 0:
#         reg_coef = [np.nan, np.nan, np.nan, np.nan]
#     else:
#         reg = LinearRegression().fit(df_use, y)
#         reg_coef = reg.coef_
#
#     for val in zip(df_use.columns, reg_coef):
#         if ns == 0:
#             val_reg[val[0]] = []
#         val_reg[val[0]].append(val[1])
#
# val_reg['speed_bins'] = speed_bins
# pickle.dump(val_reg, open('pickle/save_regs_' + comp_class_use+'.p', 'wb'))

# Some code we might need. Subject to refactoring

# fig, ax = plt.subplots(2, 2)
# ax = ax.flatten()
# for nv, val in enumerate(df_use.columns):
#     ax[nv].plot(speed_bins, val_reg[val])
#     ax[nv].set_ylabel(r'$\Delta$ Task Speed Top 3/ $\Delta$ '+val, fontsize=12)
#     ax[nv].set_xlabel('Task Speed Top 3 (km/h)')
# plt.show()

# print(np.mean(df_abs['Task speed']))
# X_train, X_test, y_train, y_test = train_test_split(df_use, y, test_size=0.01, random_state=74)

# vif_data = pd.DataFrame()
# vif_data["feature"] = X_train.columns
# # calculating VIF for each feature
# vif_data["VIF"] = [variance_inflation_factor(X_train.values, i)
#                           for i in range(len(X_train.columns))]
# print(vif_data)

# reg = LinearRegression().fit(X_train, y_train)
# print('R^2='+str(reg.score(X_train, y_train)))
# for val in zip(X_train.columns, reg.coef_):
#     print(str(val[0])+": "+str(val[1]))

# y_predict = reg.predict(X_test)
# plt.plot(y_test, y_test, c='black')
# plt.scatter(y_test, y_predict, c='red', s=3)
# plt.xlabel('Actual speed difference to Top 3 (km/h)')
# plt.ylabel('Predicted speed difference to Top 3 (km/h)')
# plt.grid(True)
# plt.show()

# df_use = df_abs.drop('Distance covered from task', axis=1)
# for nv1, var1 in enumerate(df_use.columns):
#     var_name_plot = var1
#     if var1 == "Average L/D":
#         var_name_plot = var1.replace('L/D', 'LD')
#     pdf_pages = PdfPages('correlation/pysoar_'+var_name_plot+'_correlation.pdf')
#     for nv2, var2 in enumerate(df_use.columns):
#         fig = plt.figure()
#         plt.scatter(df_use[var2], df_use[var1], c='red', s=1)
#         plt.ylabel('$\Delta$ '+var1+' '+var_units[var1])
#         plt.xlabel('$\Delta$ '+var2+' '+var_units[var2])
#         reg = LinearRegression().fit(df_use[[var2]], df_use[var1])
#         m, r2 = np.around(reg.coef_[0], 3), np.around(reg.score(df_use[[var2]], df_use[var1]), 3)
#         x_reg = np.linspace(np.min(df_use[[var2]]), np.max(df_use[[var2]]), 100)
#         y_reg = reg.intercept_+x_reg*reg.coef_
#         plt.plot(x_reg, y_reg, c='black', ls='--')
#         pdf_pages.savefig(fig)
#         plt.close(fig)
#     pdf_pages.close()
#
#
# corr_df = df_use.corr()
# fig, ax_corr = plt.subplots(figsize=(15, 15))
# matrix = np.triu(corr_df)
# sn.heatmap(corr_df, ax=ax_corr, annot=True, cmap="coolwarm", center=0)
# plt.yticks(fontsize=18)
# plt.xticks(fontsize=18)
# plt.tight_layout()
# plt.savefig('correlation/pysoar_correlation.pdf')
#
# exit()
