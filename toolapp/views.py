from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from django.views.generic import ListView
from django.http import HttpResponseRedirect, HttpResponse

from .forms import DatasetForm, DataModelForm, RegistrationForm, ChooseColumnsForm
from .models import Dataset, DataModel, Result

import pandas as pd, numpy as np, os, json, io, base64
# from PIL import Image
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold
# import matplotlib.pyplot as plt
# import seaborn as sns; sns.set(style='white')


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
            else:
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
    context = {'data_table': df_html, 'title': dataset.title, 'description': dataset.description, 
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
    if request.method == 'POST':
        form = ChooseColumnsForm(request.POST)
        if form.is_valid():
            dataset.sampleColumn = form.cleaned_data['sampleColumn']
            dataset.classColumn = form.cleaned_data['classColumn']
            dataset.featureColumns = pd.Series(form.cleaned_data['features']).to_json(orient='values')
            dataset.save()
            return HttpResponseRedirect("/toolapp/dataset/" + str(pk))
        return render(request, 'change_columns.html', {'form': form, 'columns': columns, 'features': features[0], 'sample': sampleColumn, 'class': classColumn})
    else:
        form = ChooseColumnsForm()
        return render(request, 'change_columns.html', {'form': form, 'columns': columns, 'features': features[0], 'sample': sampleColumn, 'class': classColumn})

# def drawGraphic(X, y = ""):
#     fig = plt.figure()
#     if (type(y) == str):
#         plt.scatter(X[:, 0], X[:, 1])
#     else:
#         classes = y.unique()
#         for i in range (0, classes.size):
#             plt.scatter(X[y == classes[i], 0], X[y == classes[i], 1], label = classes[i])
#     plt.legend()
#     canvas = fig.canvas
#     buf, size = canvas.print_to_buffer()
#     image = Image.frombuffer('RGBA', size, buf, 'raw', 'RGBA', 0, 1)
#     buffer=io.BytesIO()
#     image.save(buffer,'PNG')
#     graphic = buffer.getvalue()
#     graphic = base64.b64encode(graphic)
#     buffer.close()
#     return graphic

@login_required
def datasetPCAView(request, pk):
    dataset = Dataset.objects.get(pk=pk)
    df = pd.read_json(dataset.df).sort_index()
    features = pd.read_json(dataset.featureColumns)
    X_normalized = pd.DataFrame(preprocessing.normalize(preprocessing.scale(df[np.array(features[0])])),
      columns = features)
    X = df[np.array(features[0])]
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X)
    X_normalized_pca = pca.fit_transform(X_normalized)
    y = df[dataset.classColumn] if (dataset.classColumn != "") else ""
    graphic = drawGraphic(X_pca, y)
    graphic_normalized = drawGraphic(X_normalized_pca, y)
    return render(request, 'pca.html',{'graphic': str(graphic)[2:-1], 'graphic_normalized': str(graphic_normalized)[2:-1], 'title': dataset.title, 'description': dataset.description,
    'owner': dataset.owner, 'pk': dataset.pk, 'date': dataset.date})

@login_required
def createModelView(request, pk):
    if request.method == 'POST':
        form = DataModelForm(request.POST)
        if form.is_valid():
            dataset = Dataset.objects.get(pk = pk)
            df = pd.read_json(dataset.df).sort_index()
            features = pd.read_json(dataset.featureColumns)
            X = pd.DataFrame(preprocessing.normalize(preprocessing.scale(df[np.array(features[0])])),
                columns = features)
            resultDf = pd.DataFrame(df[np.array(features[0])])
            if (dataset.classColumn != ""):
                resultDf.insert(0, dataset.classColumn, df[dataset.classColumn])
            if (dataset.sampleColumn != ""):
                resultDf.insert(0, dataset.sampleColumn, df[dataset.sampleColumn])        
            if (form.cleaned_data['modelType'] == "Кластеризация методом k-средних"):
                numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
                X = X[numeric_columns].fillna(X[numeric_columns].mean())
                best_k = 2
                best_score = -1
                clusters = np.zeros(X.shape[0])
                for k in range (form.cleaned_data["minClusters"], form.cleaned_data["maxClusters"]):
                    model = KMeans(k, random_state = 0).fit(X)
                    curr_score = silhouette_score(X, model.labels_)
                    if (curr_score > best_score):
                        best_k = k
                        best_score = curr_score
                        clusters = model.labels_
                resultDf.insert(0,'Кластер', clusters)
                score = best_score
            elif (form.cleaned_data['modelType'] == "Классификация деревом решений"):
                y = pd.DataFrame(df[dataset.classColumn])
                numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
                other_columns = X.select_dtypes(exclude=np.number).columns.tolist()
                X[numeric_columns].fillna(X[numeric_columns].mean(),inplace = True)
                if (other_columns):
                    X[other_columns].fillna(X[other_columns].mode().iloc[0],inplace = True)
                (unique, counts) = np.unique(y, return_counts=True)
                tree = DecisionTreeClassifier()
                tree_parameters = {'min_samples_leaf': range(2, 5),
                   'max_depth': range(2, X.shape[0] ),
                   'max_features': range(2, features.size + 1)}
                if (form.cleaned_data["parameterSearchMethod"] == "Randomized search"):
                    tree_grid = RandomizedSearchCV(tree, tree_parameters,cv=counts.min(), n_jobs=-1,verbose=False)
                else:
                     tree_grid = GridSearchCV(tree, tree_parameters,cv=counts.min(), n_jobs=-1,verbose=False)
                tree_grid.fit(X, y)       
                skf = StratifiedKFold(n_splits=counts.min())
                skf.split(X, y)
                tree = tree_grid.best_estimator_
                result = pd.DataFrame()
                for train_index, test_index in skf.split(X, y):
                    tree.fit(X.iloc[train_index], y.iloc[train_index])
                    predictions = pd.DataFrame(tree.predict(X.iloc[test_index]), test_index, columns = ['Определенный_моделью_класс'])
                    result = result.append(predictions)
                resultDf = result.join(resultDf).sort_index()
                score = accuracy_score(y, resultDf['Определенный_моделью_класс'])       
            resultJSON = resultDf.to_json()
            new_model = DataModel.objects.create(dataset = dataset, modelType = form.cleaned_data['modelType'])
            new_result = Result.objects.create(dataModel = new_model, df = resultJSON, score = score)
            return HttpResponseRedirect("/toolapp/result/" + str(new_result.pk))
        return render(request, 'create_model.html', {'form': form})
    else:
        form = DataModelForm()
        dataset = Dataset.objects.get(pk = pk)
        df = pd.read_json(dataset.df).sort_index()
        classChosen = False if (dataset.classColumn == "") else True
        return render(request, 'create_model.html', {'form': form, 'samples': df.shape[0], 'class': classChosen})

@login_required
def modelsListView(request, pk):
    dataset = Dataset.objects.get(pk = pk)
    models = DataModel.objects.filter(dataset = dataset)
    return render(request, 'model_list.html', {'dataset_pk': dataset.pk, 'models': models, 'title': dataset.title})

@login_required
def modelDetailView(request, pk):
    model = DataModel.objects.get(pk = pk)
    results = Result.objects.filter(dataModel = model)
    return render(request, 'model_detail.html', {'dataset_pk': model.dataset.pk, 'results': results, 'modelType': model.modelType, 'owner': model.dataset.owner, 'date': model.date})

@login_required
def resultDetailView(request, pk):
    result = Result.objects.get(pk=pk)
    dataset = result.dataModel.dataset
    model_pk = result.dataModel.pk
    df = pd.read_json(result.df).sort_index()
    df_html = df.to_html()
    context = {'result_pk': pk, 'model_pk': model_pk, 'data_table': df_html, 'title': dataset.title, 'description': dataset.description, 
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
    

