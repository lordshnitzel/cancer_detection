import pickle
import numpy as np
import joblib
from django.http import HttpResponse
from django.shortcuts import render


def home(request):
    return render(request, 'home.html')


def result(request,):
    m_path = ('/home/shnitzel/PycharmProjects/breast_project/DeployModel/model1.pkl')

    cls = pickle.load(open(m_path, 'rb'))

    lis= []

    lis.append(request.GET['clump_thickness'])
    lis.append(request.GET['uniform_cell_size'])
    lis.append(request.GET['uniform_cell_shape'])
    lis.append(request.GET['marginal_adhesion'])
    lis.append(request.GET['single_epithelial_size'])
    lis.append(request.GET['bland_chromatin'])
    lis.append(request.GET['normal_nucleoli'])
    lis.append(request.GET['mitoses'])
    lis.append(request.GET['class'])

    arr_list=np.array(lis)
    arr_list=arr_list.reshape(1,-1)
    print([arr_list])

    ans = cls.predict(arr_list)

    return render(request, 'result.html', {'ans': ans})
