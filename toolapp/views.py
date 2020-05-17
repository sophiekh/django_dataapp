from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.http import HttpResponseRedirect, HttpResponse, JsonResponse

from .forms import DatasetForm, DataModelForm, RegistrationForm, ChooseColumnsForm
from .models import Dataset, DataModel, Result

import pandas as pd, numpy as np, os, json, io, base64
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
import django_rq
from django_rq import job
from redis import Redis
from rq.job import Job
from rq import get_current_job
import time

class HomePageView(LoginRequiredMixin, ListView):
    model = Dataset
    template_name = 'home.html'
    
    def get_queryset(self):
        return Dataset.objects.filter(owner=self.request.user)

@login_required
def createDatasetView(request): 
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            filename, extension = os.path.splitext(request.FILES['data'].name)
            if (extension == '.csv'):
                df = pd.read_csv(request.FILES['data'])
            elif (extension == '.xlsx'):
                df = pd.read_excel(request.FILES['data'])
            columns = df.columns
            dfJSON = df.to_json()
            featureColumns = pd.Series(columns[2:]).to_json(orient='values')
            classColumn = columns[1]
            sampleColumn = columns[0]
            new_dataset = Dataset.objects.create(title = form.cleaned_data['title'], 
            description = form.cleaned_data['description'], df = dfJSON, classColumn = classColumn, 
            sampleColumn = sampleColumn, featureColumns = featureColumns, owner = request.user)
            return HttpResponseRedirect("/toolapp/dataset/" + str(new_dataset.pk))
        return render(request, 'dataset.html', {'form': form})
    else:
        form = DatasetForm()
        return render(request, 'dataset.html', {'form': form})

@login_required
def deleteDatasetView(request, pk):
    dataset = Dataset.objects.get(pk = pk)
    if (request.user == dataset.owner):
        dataset.delete()
    return HttpResponseRedirect("/toolapp/")

@login_required
def deleteModelView(request, pk):
    model = DataModel.objects.get(pk = pk)
    dataset_pk = model.dataset.pk
    if (request.user == model.dataset.owner):
        model.delete()
    return HttpResponseRedirect("/toolapp/dataset/" + str(dataset_pk) + "/models/")

@login_required
def datasetDetailView(request, pk):
    dataset = Dataset.objects.get(pk=pk)
    df = pd.read_json(dataset.df).sort_index()
    features = pd.read_json(dataset.featureColumns)
    features_list = features[0]
    sampleColumn = dataset.sampleColumn
    classColumn = dataset.classColumn
    df_html = df.to_html()
    context = {'data_table': df_html, 'pk': dataset.pk, 'title': dataset.title, 'description': dataset.description, 
    'owner': dataset.owner, 'date': dataset.date, 'sample': sampleColumn, 'class': classColumn, 'features_list': features_list}
    return render(request, "dataset_detail.html", context)

@login_required
def changeColumnsView(request, pk):
    dataset = Dataset.objects.get(pk = pk)
    df = pd.read_json(dataset.df).sort_index()
    features = pd.read_json(dataset.featureColumns)
    sampleColumn = dataset.sampleColumn
    classColumn = dataset.classColumn
    columns = set(df.columns) - set(features[0]) - {sampleColumn} - {classColumn}
    context = {'columns': columns, 'features': features[0], 'sample': sampleColumn,
        'class': classColumn, 'pk': dataset.pk, 'title': dataset.title}
    if request.method == 'POST':
        form = ChooseColumnsForm(request.POST)
        if form.is_valid():
            dataset.sampleColumn = form.cleaned_data['sampleColumn']
            dataset.classColumn = form.cleaned_data['classColumn']
            dataset.featureColumns = pd.Series(form.cleaned_data['features']).to_json(orient='values')
            dataset.save()
            return HttpResponseRedirect("/toolapp/dataset/" + str(pk))
        return render(request, 'change_columns.html', {'form': form, 'columns': columns, 'features': features[0], 'sample': sampleColumn, 'class': classColumn, 'pk': dataset.pk, 'title': dataset.title})
    else:
        form = ChooseColumnsForm()
        return render(request, 'change_columns.html', {'form': form, 'columns': columns, 'features': features[0], 'sample': sampleColumn, 'class': classColumn, 'pk': dataset.pk, 'title': dataset.title,})

def getGraphicData(request, pk):
    dataset = Dataset.objects.get(pk=pk)
    df = pd.read_json(dataset.df).sort_index()
    features = pd.read_json(dataset.featureColumns)
    X = df[np.array(features[0])]
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    X = X[numeric_columns].fillna(X[numeric_columns].mean())
    X_normalized = pd.DataFrame(preprocessing.normalize(preprocessing.scale(X)),
      columns = X.columns)
    y = df[dataset.classColumn] if (dataset.classColumn != "") else ""
    data = {'standard': makeData(X, y, dataset.title), 'normalized': makeData(X_normalized, y, dataset.title)}
    data = json.dumps(data)
    return JsonResponse(data, safe = False)

def makeData(X, y, title):
    pca = PCA(n_components = 2)
    pca.fit(X)
    X_pca = pd.DataFrame(pca.transform(X), columns=['x', 'y'], index=X.index)
    if (type(y) == str):
        data = {'datasets' : [{'label': title, 'data': X_pca.to_dict(orient = "records")}]}
    else:
        data = {'datasets': []}
        unique = y.unique()
        if (y.isnull().values.any()):
            missing_data = X_pca[y.isnull()].to_dict(orient = 'records')
            missing_dict = {'label': 'Не определен', 'data' : missing_data}
            data['datasets'].append(missing_dict)
            unique = unique[~pd.isnull(unique)]
        for i in range(0, len(unique)):
            class_data = X_pca[y == unique[i]].to_dict(orient = 'records')
            class_dict = {'label': str(unique[i]), 'data' : class_data}
            data['datasets'].append(class_dict)
    return data

@login_required
def datasetPCAView(request, pk):
    dataset = Dataset.objects.get(pk=pk)
    return render(request, 'pca.html',{'title': dataset.title, 'description': dataset.description,
    'owner': dataset.owner, 'pk': dataset.pk, 'date': dataset.date})

def dbscan_score(X, clusters):
    if (len(np.unique(clusters))>1):
        return silhouette_score(X, clusters)
    else:
        return -1

def buildModel(dataset_pk, model_pk, result_pk, modelType, parameters):
    dataset = Dataset.objects.get(pk = dataset_pk)
    new_model = DataModel.objects.get(pk = model_pk)
    new_result = Result.objects.get(pk = result_pk)
    try:
        df = pd.read_json(dataset.df).sort_index()
        features = pd.read_json(dataset.featureColumns)
        X = df[np.array(features[0])]
        numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
        X = X[numeric_columns].fillna(X[numeric_columns].mean())
        X = pd.DataFrame(preprocessing.normalize(preprocessing.scale(X)),
        columns = X.columns)
        resultDf = pd.DataFrame(df[np.array(features[0])])
        score = 0.0
        if (dataset.classColumn != ""):
            resultDf.insert(0, dataset.classColumn, df[dataset.classColumn])
        if (dataset.sampleColumn != ""):
            resultDf.insert(0, dataset.sampleColumn, df[dataset.sampleColumn])    
        if (modelType == "Кластеризация методом k-средних"):
            best_k = 2
            best_score = -1
            clusters = np.zeros(X.shape[0])
            for k in range (parameters['minClusters'], parameters["maxClusters"]):
                model = KMeans(k, random_state = 0).fit(X)
                curr_score = silhouette_score(X, model.labels_)
                if (curr_score > best_score):
                    best_k = k
                    best_score = curr_score
                    clusters = model.labels_
            resultDf.insert(0,'Кластер', clusters)
            score = best_score
        elif (modelType == "Кластеризация методом DBSCAN"):
            neigh = NearestNeighbors(n_neighbors=2)
            nbrs = neigh.fit(X)
            distances, indices = nbrs.kneighbors(X)
            distances = distances[:,1]
            if (distances.min() == 0.0):
                eps = np.linspace(0.001,distances.max(),30)
            else:
                eps = np.linspace(distances.min(),distances.max(),30)
            min_samples = range(2, 15)
            clusters = np.zeros(X.shape[0])
            best_score = -1
            for e in eps:
                for s in min_samples:
                    model = DBSCAN(eps = e, min_samples = s)
                    model.fit(X)
                    curr_score = dbscan_score(X, model.labels_)
                    if (curr_score > best_score):
                        best_score = curr_score
                        clusters = model.labels_
            resultDf.insert(0,'Кластер', clusters)
            score = best_score
        elif (modelType == "Классификация деревом решений"):
            X['class'] = pd.DataFrame(df[dataset.classColumn])
            test = X.loc[X.isnull()['class']].drop(columns = ['class'])
            train = X.loc[~X.isnull()['class']].drop(columns = ['class'])
            train_y = pd.DataFrame(df[dataset.classColumn]).dropna()
            (unique, counts) = np.unique(train_y, return_counts=True)
            tree = DecisionTreeClassifier()
            tree_parameters = {'min_samples_leaf': range(2, 5),
                'max_depth': range(2, train.shape[0]),
                'max_features': range(2, features.size + 1)}
            cv = counts.min() if counts.min() < 5 else 5
            if (parameters["parameterSearchMethod"] == "Случайный поиск"):
                tree_grid = RandomizedSearchCV(tree, tree_parameters,cv=cv, n_jobs=-1,verbose=False)
            else:
                tree_grid = GridSearchCV(tree, tree_parameters,cv=cv, n_jobs=-1,verbose=False)
            tree_grid.fit(train, train_y)       
            score = tree_grid.best_score_
            predictions = pd.DataFrame(tree_grid.predict(test), test.index, columns = ['Определенный_моделью_класс'])
            resultDf = predictions.join(resultDf, how = 'outer')
        elif (modelType == "Классификация случайным лесом"):
            X['class'] = pd.DataFrame(df[dataset.classColumn])
            test = X.loc[X.isnull()['class']].drop(columns = ['class'])
            train = X.loc[~X.isnull()['class']].drop(columns = ['class'])
            train_y = pd.DataFrame(df[dataset.classColumn]).dropna()
            (unique, counts) = np.unique(train_y, return_counts=True)
            forest = RandomForestClassifier()
            random_parameters = {'n_estimators': [100, 250, 500],
                        'max_features': ['auto', 'sqrt'],
                        'max_depth': [int(x) for x in np.linspace(2, 20, num = 2)],
                        'min_samples_split': [2, 3, 4, 5],
                        'min_samples_leaf': [2, 3, 4]}
            cv = counts.min() if counts.min() < 5 else 5
            if (parameters["parameterSearchMethod"] == "Поиск по сетке"):
                random_grid = GridSearchCV(forest, random_parameters,cv=cv, n_jobs=-1,verbose=False)
            else:
                    random_grid = RandomizedSearchCV(forest, random_parameters,cv=cv, n_jobs=-1,verbose=False)
            random_grid.fit(train, train_y.values.ravel())       
            score = random_grid.best_score_
            predictions = pd.DataFrame(random_grid.predict(test), test.index, columns = ['Определенный_моделью_класс'])
            resultDf = predictions.join(resultDf, how = 'outer')      
        resultJSON = resultDf.to_json()
        new_result.df = resultJSON
        new_result.score = score
        new_result.save()
        return new_result.pk
    except:
        new_model.delete()

@login_required
def createModelView(request, pk):
    if request.method == 'POST':
        modelType = request.POST.get('modelType')
        if (modelType == "Кластеризация методом k-средних"):
            parameters = {'minClusters': int(request.POST.get("minClusters")), 'maxClusters': int(request.POST.get("maxClusters"))}
        elif (modelType == "Кластеризация методом DBSCAN"):
            parameters = {}
        elif ((modelType == "Классификация деревом решений")or (modelType == "Классификация случайным лесом")):
            parameters = {'parameterSearchMethod': request.POST.get("parameterSearchMethod")}
        dataset = Dataset.objects.get(pk = pk)
        new_model = DataModel.objects.create(dataset = dataset, modelType = modelType)
        new_result = Result.objects.create(dataModel = new_model, df = '', score = 0)
        model_pk = new_model.pk
        result_pk = new_result.pk
        queue = django_rq.get_queue('low')
        job = queue.enqueue(buildModel, args = (pk, model_pk, result_pk, modelType, parameters), job_timeout='60m', result_ttl=86400)    
        return JsonResponse({"job_id": job.id, 'result_pk': new_result.pk})
    else:
        form = DataModelForm()
        dataset = Dataset.objects.get(pk = pk)
        df = pd.read_json(dataset.df).sort_index()
        classChosen = False if (dataset.classColumn == "") else True
        return render(request, 'create_model.html', {'form': form, 'samples': df.shape[0], 'class': classChosen, 'dataset_pk': dataset.pk, 'title': dataset.title})

@login_required
def getJobInfo(request, id):
    redis = Redis()
    redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
    redis = redis.from_url(redis_url)
    job = Job.fetch(id, connection=redis)
    job_status =  job.get_status()
    return JsonResponse({'status': job_status})

@login_required
def modelsListView(request, pk):
    dataset = Dataset.objects.get(pk = pk)
    models = DataModel.objects.filter(dataset = dataset)
    return render(request, 'model_list.html', {'dataset_pk': dataset.pk, 'models': models, 'title': dataset.title})

@login_required
def modelDetailView(request, pk):
    model = DataModel.objects.get(pk = pk)
    results = Result.objects.filter(dataModel = model)
    return render(request, 'model_detail.html', {'model_pk': pk, 'dataset_pk': model.dataset.pk, 'title': model.dataset.title, 'results': results, 'modelType': model.modelType, 'owner': model.dataset.owner, 'date': model.date})

@login_required
def resultDetailView(request, pk):
    result = Result.objects.get(pk=pk)
    dataset = result.dataModel.dataset
    model_pk = result.dataModel.pk
    df = pd.read_json(result.df).sort_index()
    df_html = df.to_html()
    context = {'dataset_pk': dataset.pk, 'result_pk': pk, 'model_pk': model_pk, 'data_table': df_html, 'title': dataset.title, 'description': dataset.description, 
    'owner': dataset.owner, 'model_date': result.date, 'type': result.dataModel.modelType, 'date': dataset.date, 'score': result.score}
    return render(request, "result_detail.html", context)

@login_required
def resultDownloadView(request, pk):
    result = Result.objects.get(pk=pk)
    df = pd.read_json(result.df).sort_index()
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=result_' + str(pk) + '.xlsx'
    df.to_excel(excel_writer=response,float_format='%.2f', index = False)
    return response

def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect("home")
        else:
            return render(request = request,
                          template_name = "register.html",
                          context={"form":form})
    form = UserCreationForm
    return render(request = request,
                  template_name = "register.html",
                  context={"form":form})
    

