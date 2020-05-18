from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
  title = models.CharField(max_length=100, help_text="Название набора данных", verbose_name="Название")
  description = models.TextField(max_length=250, help_text="Описание набора данных", verbose_name="Описание")
  df = models.TextField(default = '', verbose_name = "DataFrame")
  classColumn = models.TextField(default = '', verbose_name = "Class")
  sampleColumn = models.TextField(default = '', verbose_name = "Sample")
  featureColumns = models.TextField(default = '', verbose_name = "Features")
  owner = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
  date = models.DateTimeField(auto_now_add = True)

  def __str__(self):
        return  '{0}: {1}'.format (self.title, self.description)

class DataModel(models.Model):
      dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True, blank=True)
      modelType = models.CharField(max_length=250)
      info = models.TextField(default = '')
      date = models.DateTimeField(auto_now_add = True)

class Result(models.Model):
      dataModel = models.OneToOneField(DataModel, on_delete=models.CASCADE, primary_key=True)
      df = models.TextField(default = '')
      score = models.DecimalField(default = 0.0, max_digits = 4, decimal_places = 2)






  

