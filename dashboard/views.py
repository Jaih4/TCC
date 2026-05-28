import os
import re as regex
from datetime import datetime
from typing import Any, Dict, List
from collections import Counter
import json
from api.models import Comment
from django.conf import settings
from django.shortcuts import render
from dashboard.forms import InstagramDashboardForm
import pandas as pd
from .forms import PostEspecificoForm
from analise_nlp.instagram_pipeline import (
    ApifyInstagramScraper,
    processar_comentarios,
    salvar_resultados_no_banco
)

def analisar_post_especifico_view(request):
    # Contexto inicial padrão para requisições GET
    form = PostEspecificoForm()
    context = {
        'form': PostEspecificoForm(),
        'submitted': False,
        'error_message': None,
    }

    if request.method == 'POST':
        form = PostEspecificoForm(request.POST)
        context['form'] = form
        context['submitted'] = True
        
        if form.is_valid():
            post_url = form.cleaned_data['post_url']
            data_referencia = form.cleaned_data['data_referencia']
            
            # Removemos espaços em branco e um eventual '@' caso o usuário digite
            profile_name = form.cleaned_data['profile_name'].strip().lstrip('@')
            
            try:
                # 1. Instancia o scraper (Substitua pela sua forma de obter o token)
                token = getattr(settings, 'APIFY_TOKEN', 'SEU_TOKEN_APIFY_AQUI')
                scraper = ApifyInstagramScraper(
                    apify_token=token,
                    max_comments_per_post=200 
                )
                
                # 2. Chama o método que vai direto para o post e filtra pela data
                filtered_comments = scraper.scrape_comments_for_post(post_url, data_referencia)
                
                # Se a lista voltar vazia, encerra a execução e avisa o HTML
                if not filtered_comments:
                    context['total_comments'] = 0
                    return render(request, 'dashboard/post_dashboard.html', context)
                    
                # 3. Passa os comentários filtrados pelo pipeline de NLP
                df = processar_comentarios(filtered_comments)
                context['total_comments'] = len(df)
                
                # 4. Salva no banco de dados
                # Passamos o nome do perfil informado no form para o DB
                salvar_resultados_no_banco(df, profile_handle=profile_name)
                context['saved_table'] = True

                # ==========================================
                # PREPARAÇÃO DOS DADOS PARA RENDERIZAÇÃO NO HTML
                # ==========================================
                
                # A. Estatísticas de Sentimento (Baseado no score_p)
                apoio = len(df[df['score_p'] > 0])
                rejeicao = len(df[df['score_p'] < 0])
                neutro = len(df[df['score_p'] == 0])
                total = len(df)
                
                context['stats_data'] = [
                    {
                        'label': 'Apoio / Positivo', 
                        'count': apoio, 
                        'pct_str': round((apoio / total) * 100, 1) if total > 0 else 0, 
                        'color': 'success'
                    },
                    {
                        'label': 'Rejeição / Negativo', 
                        'count': rejeicao, 
                        'pct_str': round((rejeicao / total) * 100, 1) if total > 0 else 0, 
                        'color': 'danger'
                    },
                    {
                        'label': 'Neutro', 
                        'count': neutro, 
                        'pct_str': round((neutro / total) * 100, 1) if total > 0 else 0, 
                        'color': 'secondary'
                    }
                ]
                
                # B. Top 20 Palavras
                # Junta todo o texto, normaliza para minúsculas e extrai palavras >= 3 letras
                texto_completo = " ".join(df['texto'].dropna().astype(str)).lower()
                palavras = re.findall(r'\b[a-zà-ú]{3,}\b', texto_completo)
                
                # Filtro de stopwords (você pode expandir essa lista depois)
                stopwords = {
                    'que', 'para', 'com', 'não', 'uma', 'dos', 'das', 'por', 
                    'mais', 'como', 'mas', 'foi', 'ele', 'ela', 'isso', 'esse',
                    'este', 'pra', 'pro', 'aos', 'nas', 'nos'
                }
                palavras_uteis = [p for p in palavras if p not in stopwords]
                
                # O HTML espera uma tupla (palavra, frequencia)
                context['top_words'] = Counter(palavras_uteis).most_common(20)
                
                # C. Detalhes dos Comentários (Tabela)
                # Seleciona as colunas, ordena pelo maior score e converte para dict pro Django iterar
                df_results = df[['score_p', 'texto']].sort_values(by='score_p', ascending=False)
                context['results'] = df_results.to_dict('records')

            except Exception as e:
                # Se algo quebrar no meio do processo, captura o erro e envia pro HTML
                context['error_message'] = f"Ocorreu um erro na extração/análise: {str(e)}"
                
    return render(request, 'dashboard/post_dashboard.html', context)

def _normalize_instagram_profile(profile_or_url: str) -> str:
    raw = (profile_or_url or '').strip()
    if raw.startswith('http'):
        # ANTES: match = re.search(...)
        match = regex.search(r'(?:instagram\.com/(?:p/|tv/|reel/)?@?)([A-Za-z0-9_.-]+)', raw)
        if match:
            return match.group(1)
    if raw.startswith('@'):
        return raw[1:]
    return raw


def dashboard(request):
    form = InstagramDashboardForm(request.POST or None)
    results: List[Dict[str, Any]] = []
    error_message = None
    stats_data = []
    top_words = []
    total_comments = 0

    if request.method == 'POST' and form.is_valid():
        profile_url = form.cleaned_data['profile_url']
        data_referencia = form.cleaned_data['data_referencia']
        profile_handle = _normalize_instagram_profile(profile_url)

        apify_token = getattr(settings, 'APIFY_TOKEN', None) or os.environ.get('APIFY_TOKEN')
        if not apify_token:
            error_message = (
                'O token APIFY_TOKEN não está definido. Defina em settings.py ou na variável de ambiente APIFY_TOKEN.'
            )
        else:
            try:
                scraper = ApifyInstagramScraper(
                    apify_token=apify_token,
                    max_posts=20,
                    max_comments_per_post=200,
                )
                comments = scraper.scrape_comments_for_profile(profile_handle, data_referencia)
                df = processar_comentarios(comments)
                results = df.to_dict('records') if not df.empty else []
                if df is not None and not df.empty:
                    print(df)
                    saved_table = salvar_resultados_no_banco(df, profile_handle)
                else:
                    saved_table = None
                
                # --- LÓGICA DO DASHBOARD DE APOIO E CRÍTICA ---
                total_comments = len(results)
                stats = {
                    'Muito Ofensivo': 0,
                    'Ofensivo': 0,
                    'Neutro': 0,
                    'Favorável': 0,
                    'Muito Favorável': 0,
                }
                word_counter = Counter()
                # Stopwords para evitar que preposições e pronomes dominem os termos frequentes
                stopwords = {'meu', 'que', 'por', 'para', 'com', 'uma', 'um', 'não', 'sim', 'isso', 'mais', 'como', 'mas', 'foi', 'aos', 'nas', 'das', 'dos', 'pra', 'nos', 'tem', 'ele', 'ela', 'eles', 'elas', 'este', 'esse', 'isso', 'aquilo', 'muito', 'são', 'ser', 'está', 'quando', 'quem', 'qual', 'seu', 'sua', 'você', 'sobre', 'também', 'pelo', 'pela', 'nem', 'até', 'sem'}

                for row in results:
                    score = row.get('score_p', 0)
                    try:
                        score = int(score)
                    except (ValueError, TypeError):
                        score = 0
                        
                    if score <= -4:
                        stats['Muito Ofensivo'] += 1
                    elif score in [-3, -2]:
                        stats['Ofensivo'] += 1
                    elif score in [-1, 0, 1]:
                        stats['Neutro'] += 1
                    elif score in [2, 3]:
                        stats['Favorável'] += 1
                    elif score >= 4:
                        stats['Muito Favorável'] += 1
                        
                    # Capturar palavras com mais de 3 caracteres
                    texto = str(row.get('texto', '')).lower()
                    # ANTES: palavras = re.findall(...)
                    palavras = regex.findall(r'\b[a-zà-ú]{3,}\b', texto)
                    for p in palavras:
                        if p not in stopwords:
                            word_counter[p] += 1

                if total_comments > 0:
                    stats_pct = {k: (v / total_comments * 100) for k, v in stats.items()}
                else:
                    stats_pct = {k: 0 for k in stats.keys()}

                # Preparando a estrutura final em lista para facilitar a visualização no HTML
                stats_data = [
                    {'label': 'Muito Favorável (>= 4)', 'count': stats['Muito Favorável'], 'pct_str': str(round(stats_pct['Muito Favorável'], 1)).replace(',', '.'), 'pct_val': round(stats_pct['Muito Favorável'], 1), 'color': 'success'},
                    {'label': 'Favorável (2 a 3)', 'count': stats['Favorável'], 'pct_str': str(round(stats_pct['Favorável'], 1)).replace(',', '.'), 'pct_val': round(stats_pct['Favorável'], 1), 'color': 'info'},
                    {'label': 'Neutro (-1 a 1)', 'count': stats['Neutro'], 'pct_str': str(round(stats_pct['Neutro'], 1)).replace(',', '.'), 'pct_val': round(stats_pct['Neutro'], 1), 'color': 'secondary'},
                    {'label': 'Ofensivo (-3 a -2)', 'count': stats['Ofensivo'], 'pct_str': str(round(stats_pct['Ofensivo'], 1)).replace(',', '.'), 'pct_val': round(stats_pct['Ofensivo'], 1), 'color': 'warning'},
                    {'label': 'Muito Ofensivo (<= -4)', 'count': stats['Muito Ofensivo'], 'pct_str': str(round(stats_pct['Muito Ofensivo'], 1)).replace(',', '.'), 'pct_val': round(stats_pct['Muito Ofensivo'], 1), 'color': 'danger'},
                ]
                top_words = word_counter.most_common(20)

            except Exception as exc:
                error_message = str(exc)

    context = {
        'form': form,
        'results': results,
        'error_message': error_message,
        'submitted': request.method == 'POST',
        'saved_table': saved_table if 'saved_table' in locals() else None,
        'stats_data': stats_data,
        'top_words': top_words,
        'total_comments': total_comments,
    }
    return render(request, 'dashboard/index.html', context)

def database_dashboard(request):
    # Filtro opcional por nome da conta (ex: ?account=fulano)
    account_filter = request.GET.get('account')
    if account_filter:
        comments = Comment.objects.filter(account_name=account_filter)
    else:
        comments = Comment.objects.all()

    total_comments = comments.count()

    stats = {
        'Muito Ofensivo': 0,
        'Ofensivo': 0,
        'Neutro': 0,
        'Favorável': 0,
        'Muito Favorável': 0,
    }

    for c in comments:
        score = c.score
        if score <= -4:
            stats['Muito Ofensivo'] += 1
        elif -3 <= score <= -2:
            stats['Ofensivo'] += 1
        elif -1 <= score <= 1:
            stats['Neutro'] += 1
        elif 2 <= score <= 3:
            stats['Favorável'] += 1
        elif score >= 4:
            stats['Muito Favorável'] += 1

    stats_pct = {}
    for k, v in stats.items():
        stats_pct[k] = round((v / total_comments * 100), 1) if total_comments > 0 else 0

    # Obter contas únicas para o select do filtro (remove nulos e vazios)
    accounts = Comment.objects.exclude(account_name__isnull=True).exclude(account_name='').values_list('account_name', flat=True).distinct()

    context = {
        'total_comments': total_comments,
        'stats': stats,
        'stats_pct': stats_pct,
        'labels_json': json.dumps(list(stats.keys())),
        'counts_json': json.dumps(list(stats.values())),
        'accounts': accounts,
        'selected_account': account_filter,
    }
    return render(request, 'dashboard/database_dashboard.html', context)

#gpt

from django.db.models import Count, Avg
from django.db.models.functions import TruncDate
from django.shortcuts import render
from collections import Counter
import json
import re


def dashboard_nlp(request):

    account = request.GET.get('account')

    comments = Comment.objects.all()

    if account:
        comments = comments.filter(account_name=account)

    total_comments = comments.count()

    # =========================
    # ESTATÍSTICAS
    # =========================

    stats = {
        "Hate": comments.filter(hate=True).count(),
        "Ofensivo": comments.filter(bad=True).count(),
        "Positivo": comments.filter(nice=True).count(),
        "Spam": comments.filter(spam=True).count(),
        "Caixa Alta": comments.filter(caixa_alta=True).count(),
        "MiddleList": comments.filter(middlelist=True).count(),
    }

    stats_pct = {}

    for k, v in stats.items():
        stats_pct[k] = round((v / total_comments) * 100, 2) if total_comments else 0

    # =========================
    # APOIO / REJEIÇÃO
    # =========================

    apoio = comments.filter(nice=True).count()

    rejeicao = (
        comments.filter(hate=True).count() +
        comments.filter(bad=True).count()
    )

    apoio_pct = round((apoio / total_comments) * 100, 2) if total_comments else 0

    rejeicao_pct = round((rejeicao / total_comments) * 100, 2) if total_comments else 0

    # =========================
    # SCORE MÉDIO
    # =========================

    media_score = comments.aggregate(
        Avg('score_p')
    )['score_p__avg'] or 0

    media_score = round(media_score, 2)

    # =========================
    # TIMELINE
    # =========================

    timeline = comments.annotate(
        dia=TruncDate('data')
    ).values('dia').annotate(
        total=Count('id')
    ).order_by('dia')

    timeline_labels = []
    hate_timeline = []
    bad_timeline = []
    nice_timeline = []
    spam_timeline = []

    for item in timeline:

        dia = item['dia']

        timeline_labels.append(str(dia))

        hate_timeline.append(
            comments.filter(data__date=dia, hate=True).count()
        )

        bad_timeline.append(
            comments.filter(data__date=dia, bad=True).count()
        )

        nice_timeline.append(
            comments.filter(data__date=dia, nice=True).count()
        )

        spam_timeline.append(
            comments.filter(data__date=dia, spam=True).count()
        )

    # =========================
    # WORD CLOUD
    # =========================

    textos = comments.values_list('texto', flat=True)

    stopwords = {
        'de','da','do','e','o','a','que','em',
        'um','uma','para','com','não','na',
        'no','os','as','mais','mas'
    }

    palavras = []

    for texto in textos:

        if not texto:
            continue

        texto = texto.lower()

        tokens = re.findall(r'\w+', texto)

        for token in tokens:

            if len(token) > 2 and token not in stopwords:
                palavras.append(token)

    frequencia = Counter(palavras)

    wordcloud_data = [
        [k, v]
        for k, v in frequencia.most_common(120)
    ]

    context = {

        'accounts': Comment.objects.values_list(
            'account_name',
            flat=True
        ).distinct(),

        'selected_account': account,

        'total_comments': total_comments,

        'stats': stats,
        'stats_pct': stats_pct,

        'apoio_pct': apoio_pct,
        'rejeicao_pct': rejeicao_pct,

        'media_score': media_score,

        'labels_json': json.dumps(list(stats.keys())),
        'counts_json': json.dumps(list(stats.values())),

        'timeline_labels': json.dumps(timeline_labels),
        'hate_timeline': json.dumps(hate_timeline),
        'bad_timeline': json.dumps(bad_timeline),
        'nice_timeline': json.dumps(nice_timeline),
        'spam_timeline': json.dumps(spam_timeline),

        'wordcloud_data': json.dumps(wordcloud_data),
    }

    return render(
        request,
        'dashboard/dashboard_nlp.html',
        context
    )

def post(request):
    return render(request, 'dashboard/post_dashboard.html')