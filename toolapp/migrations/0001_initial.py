# Generated by Django 3.0.3 on 2020-04-15 08:27

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DataModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('modelType', models.CharField(max_length=250)),
                ('savedModel', models.TextField(default='')),
                ('date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('df', models.TextField(default='')),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('score', models.DecimalField(decimal_places=2, default=0.0, max_digits=4)),
                ('dataModel', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='toolapp.DataModel')),
            ],
        ),
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(help_text='Название набора данных', max_length=100, verbose_name='Название')),
                ('description', models.TextField(help_text='Описание набора данных', max_length=250, verbose_name='Описание')),
                ('df', models.TextField(default='', verbose_name='DataFrame')),
                ('classColumn', models.TextField(default='', verbose_name='Class')),
                ('sampleColumn', models.TextField(default='', verbose_name='Sample')),
                ('featureColumns', models.TextField(default='', verbose_name='Features')),
                ('date', models.DateTimeField(auto_now_add=True)),
                ('owner', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.AddField(
            model_name='datamodel',
            name='dataset',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='toolapp.Dataset'),
        ),
    ]
