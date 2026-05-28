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

class PostEspecificoForm(forms.Form):
    post_url = forms.URLField(
        label="Link do Post",
        help_text="Cole a URL completa do post do Instagram.",
        widget=forms.URLInput(attrs={
            'class': 'form-control',
            'placeholder': 'https://www.instagram.com/p/...'
        })
    )
    
    profile_name = forms.CharField(
        label="Nome do Perfil",
        help_text="Nome da conta para salvar no banco de dados.",
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': '@usuario'
        })
    )
    
    data_referencia = forms.DateField(
        label="Data de Referência",
        help_text="Selecione a data para filtrar os comentários.",
        widget=forms.DateInput(attrs={
            'class': 'form-control',
            'type': 'date'
        })
    )