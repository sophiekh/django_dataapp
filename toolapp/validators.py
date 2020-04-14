import os
from django.core.exceptions import ValidationError

def validate_file_extension(value):
    extension = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = [ '.xlsx', '.xls', '.csv']
    if not extension.lower() in valid_extensions:
        raise ValidationError('Неподдерживаемый формат файла.')