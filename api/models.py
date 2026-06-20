from django.db import models

class Comment(models.Model):
    # Dados básicos
    account_name = models.CharField(max_length=255, null=True, blank=True)
    texto = models.TextField(null=True, blank=True)
    data = models.DateTimeField(null=True, blank=True) # ou models.CharField se a data vier como texto puro
    like_count = models.IntegerField(null=True, blank=True)  # Número de curtidas, se disponível
    
    # Flags / Categorias (Assumindo que sejam Verdadeiro/Falso)
    hate = models.BooleanField(default=False)
    bad = models.BooleanField(default=False)
    nice = models.BooleanField(default=False)
    solida = models.BooleanField(default=False)
    caixa_alta = models.BooleanField(default=False)
    spam = models.BooleanField(default=False)
    middlelist = models.BooleanField(default=False)
    
    # Dados de Análise/IA
    modelo = models.CharField(max_length=100, null=True, blank=True)
    confianca = models.FloatField(null=True, blank=True)  # Ex: 0.95
    score_p = models.FloatField(null=True, blank=True)    # Ex: 8.5
    
    class Meta:
        # Força o Django a usar este nome exato para a tabela no banco
        db_table = 'api_comment'

    def __str__(self):
        return f"{self.account_name}: {self.texto[:30]}"
    
    from django.db import models

class TestComment(models.Model):
    account_name = models.CharField(max_length=255, null=True, blank=True)
    texto = models.TextField(null=True, blank=True)
    data = models.DateTimeField(null=True, blank=True)
    hate = models.BooleanField(default=False)
    bad = models.BooleanField(default=False)
    nice = models.BooleanField(default=False)
    solida = models.BooleanField(default=False)
    caixa_alta = models.BooleanField(default=False)
    spam = models.BooleanField(default=False)
    middlelist = models.BooleanField(default=False)
    modelo = models.CharField(max_length=255, null=True, blank=True)
    confianca = models.FloatField(default=0.0)
    score_p = models.FloatField(default=0.0)

    class Meta:
        # Força o nome exato da tabela no banco de dados
        db_table = 'test_comments'