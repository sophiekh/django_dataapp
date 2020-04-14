# Generated by Django 3.0.3 on 2020-04-11 08:42

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('toolapp', '0010_auto_20200401_1849'),
    ]

    operations = [
        migrations.AlterField(
            model_name='datamodel',
            name='dataset',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='toolapp.Dataset'),
        ),
        migrations.AlterField(
            model_name='dataset',
            name='owner',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterField(
            model_name='result',
            name='dataModel',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='toolapp.DataModel'),
        ),
    ]