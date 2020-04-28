from django.urls import path, include

from .views import HomePageView, register, getGraphicData, getJobInfo, createDatasetView, resultDownloadView, deleteDatasetView, deleteModelView, datasetDetailView, datasetPCAView, createModelView, modelsListView, modelDetailView, resultDetailView, changeColumnsView
 
 
urlpatterns = [
    path('', HomePageView.as_view(), name='home'),
    path('accounts/register/', register, name="register"),
    path('add_dataset/', createDatasetView, name='add_dataset'),
    path('delete_dataset/<int:pk>/', deleteDatasetView, name='delete_dataset'),
    path('dataset/<int:pk>/', datasetDetailView, name='dataset_detail'),
    path('dataset/<int:pk>/change_columns/', changeColumnsView, name='columns'),
    path('dataset/<int:pk>/pca/', datasetPCAView, name='pca'),
    path('dataset/<int:pk>/pca/api/data/', getGraphicData, name='data'),
    path('dataset/<int:pk>/add_model/', createModelView, name='add_model'),
    path('job_info/<str:id>/', getJobInfo, name = 'job_info'),
    path('delete_model/<int:pk>/', deleteModelView, name='delete_model'),
    path('dataset/<int:pk>/models/', modelsListView, name = 'models'),
    path('model/<int:pk>/', modelDetailView, name = 'model_detail'),
    path('result/<int:pk>/', resultDetailView, name = 'result_detail'),
    path('result/<int:pk>/download/', resultDownloadView, name = 'result_download'),
    
]

