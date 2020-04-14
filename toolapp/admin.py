from django.contrib import admin
from .models import Dataset, DataModel, Result
 
admin.site.register(Dataset)
admin.site.register(DataModel)
admin.site.register(Result)
# Register your models here.
