from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from .validators import validate_file_extension



class DatasetForm(forms.Form):
    title = forms.CharField(max_length = 100, label = "Название")
    description = forms.CharField(max_length = 250, label = "Описание")
    data = forms.FileField(label = "", validators = [validate_file_extension])




class DataModelForm(forms.Form):
    modelType = forms.ChoiceField(choices = [("Кластеризация методом k-средних", "Кластеризация методом k-средних" ),
    ("Кластеризация методом DBSCAN", "Кластеризация методом DBSCAN" ),
    ("Классификация деревом решений", "Классификация деревом решений"),
    ("Классификация случайным лесом", "Классификация случайным лесом")], label = "Тип модели")
    minClusters = forms.IntegerField()
    maxClusters = forms.IntegerField()
    parameterSearchMethod = forms.ChoiceField(choices = [("Randomized search", "Randomized search" ),
    ("Grid search", "Grid search")], label = "Метод подбора параметров модели")

class ChoiceFieldNoValidation(forms.MultipleChoiceField):
    def validate(self, value):
        pass

class ChooseColumnsForm(forms.Form):
    sampleColumn = forms.CharField(label = "Шифр образца", required = False)
    classColumn = forms.CharField(label = "Класс образца", required = False)
    features = ChoiceFieldNoValidation(choices = [], label = "Признаки", required = False)

    


class RegistrationForm(UserCreationForm):
    email = forms.EmailField(max_length=200, help_text='Required')
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')


