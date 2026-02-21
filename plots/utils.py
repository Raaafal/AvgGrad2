import ast
import numpy as np
from scipy import stats

confidence_z_score=1.#2.575829#99%

standard_error_of_the_median_mul=1.2533
def move_indexes(dict:{},move:int):
    new_dict={}
    for key in dict.keys():
        new_dict[key+move]=dict[key]
    return new_dict


def delete_where_key_is_greater_than(dict,x):
    return {k: v for k, v in dict.items() if k <= x}

# def read_file(file_name):
#     with open(file_name, 'r') as f:
#         s = f.read()
#         s=s[s.index('Best stats so far:')+len('Best stats so far:'):]
#         return ast.literal_eval(s)
def read_file(file_name):
    with open(file_name, 'r') as f:
        s = f.read()
        s=s[s.rfind('Best stats so far:')+len('Best stats so far:'):]
        return ast.literal_eval(s)

def mean_and_sem_err(arr:[]):
    return sum(arr) / len(arr), np.std(arr)/(len(arr)**0.5)
def means_and_sem_err(dict:{}):
    means={}
    err = {}
    for k, v in dict.items():
        #mean = sum(dict[k]) / len(dict[k])
        #err[k] = np.std(v) / (len(v) ** 0.5)
        #means[k]=mean

        #err[k] = np.array(v)-mean
        #err[k]=err[k]*err[k]/len(v)
        #err[k]=np.sum(err[k])/(len(v)**0.5)

        means[k],err[k]=mean_and_sem_err(v)
    return means,err

def median_and_sem_err(arr:[]):
    return np.median(arr), standard_error_of_the_median_mul*np.std(arr)/(len(arr)**0.5)
def medians_and_sem_err(dict:{}):
    medians={}
    err = {}
    for k, v in dict.items():
        medians[k],err[k]=median_and_sem_err(v)
    return medians,err

# def normality_test(statistics,threshold=0.05,verbose=True):
#     normality_test(statistics,threshold,verbose_lvl=1)
def normality_test(statistics,threshold=0.05,verbose_lvl=1):
    t_counter=0
    counter=0
    for (e,val) in statistics.items():
        test=stats.normaltest(val)
        if verbose_lvl>=2:
            print(str(e) + ':  ' + str(test))
        if test.pvalue>threshold:
            t_counter+=1
        counter+=1
    if verbose_lvl>=1:
        print('x>'+str(threshold)+':  '+str(t_counter/counter))
    return t_counter/counter

def single_normality_test(statistics):
    return stats.normaltest(statistics).pvalue

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))
def mean(iterable):
    a = np.array(iterable)
    return a.sum()/len(a)

class PlotStatistics:
    data_dict={}
    training_count=0
    #count_str=' of 0'
    method_name='Method 1'
    #label='Label 1'

    def get_label(self,stat_name='mean'):
        return self.method_name+' '+'('+stat_name+' of '+str(self.training_count)+')'
    def __init__(self,file_name):
        self.file_name=file_name

    # def __init__(self,file_name,plot_color,error_color=None):
    #     self.file_name=file_name
    #     self.plot_color=plot_color
    #     self.error_color=error_color
    #     if self.error_color is None:
    #         self.error_color=self.plot_color

    def __init__(self,method_name,file_name,plot_color,error_color=None,plot_color_median=None,error_color_median=None,linestyle=None):
        self.method_name=method_name
        self.file_name=file_name
        self.plot_color=plot_color
        self.error_color=error_color
        if self.error_color is None:
            self.error_color=self.plot_color
        self.plot_color_median=plot_color_median
        if plot_color_median is None:
            self.plot_color_median=plot_color
        self.error_color_median=error_color_median
        if error_color_median is None:
            self.error_color_median=plot_color
        self.linestyle=linestyle


def set_automatic_colors(plotstatistics_table,palette:[]):
    # colors_colorblind_safe6 = ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#D55E00", "#CC79A7"]
    colors_colorblind_safe6 = ["#E69F00","#56B4E9","#009E73","#0072B2"  , "#D55E00", "#CC79A7"]
    # colors_colorblind_safe6 = ["#E69F00", "#56B4E9","#D55E00", "#CC79A7", "#009E73", "#0072B2"]
    if not palette:
        palette=colors_colorblind_safe6
    index=0
    for plot_stats in plotstatistics_table:
        plot_stats.plot_color = palette[index]
        plot_stats.plot_color_median = palette[index]
        plot_stats.error_color = palette[index]
        plot_stats.error_color_median = palette[index]
        index+=1
        index=index%len(palette)
