from django import forms


class InstagramDashboardForm(forms.Form):
    profile_url = forms.CharField(
        required=True,
        label='Perfil ou link do Instagram',
        widget=forms.TextInput(
            attrs={
                'placeholder': '@usuario ou https://www.instagram.com/usuario',
                'class': 'form-control',
            }
        ),
    )
    data_referencia = forms.DateField(
        required=True,
        label='Data de referência',
        widget=forms.DateInput(
            attrs={
                'type': 'date',
                'class': 'form-control',
            }
        ),
    )
