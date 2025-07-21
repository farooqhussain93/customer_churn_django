from django import forms

class UploadCSVForm(forms.Form):
    csv_file = forms.FileField(label='Upload Customer Data (CSV only)', 
                               widget=forms.FileInput(attrs={'class': 'form-control'}))