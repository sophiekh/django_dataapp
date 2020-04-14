# Generated by Django 3.0.3 on 2020-03-13 16:46

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('toolapp', '0005_dataset_owner'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='dataset',
            name='data_format',
        ),
        migrations.CreateModel(
            name='Dataframe',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('df', models.TextField(verbose_name='DataFrame')),
                ('classColumn', models.TextField(verbose_name='Class')),
                ('sampleColumn', models.TextField(verbose_name='Sample')),
                ('featureColumns', models.TextField(verbose_name='Features')),
                ('dataset', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='toolapp.Dataset')),
            ],
        ),
    ]
