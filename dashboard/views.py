import os
import re
from datetime import datetime
from typing import Any, Dict, List
from collections import Counter

from django.conf import settings
from django.shortcuts import render

from analise_nlp.instagram_pipeline import (
    ApifyInstagramScraper,
    processar_comentarios,
    salvar_resultados_no_banco,
)
from dashboard.forms import InstagramDashboardForm


def _normalize_instagram_profile(profile_or_url: str) -> str:
    raw = (profile_or_url or '').strip()
    if raw.startswith('http'):
        match = re.search(r'(?:instagram\.com/(?:p/|tv/|reel/)?@?)([A-Za-z0-9_.-]+)', raw)
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
                stopwords = {'que', 'por', 'para', 'com', 'uma', 'um', 'não', 'sim', 'isso', 'mais', 'como', 'mas', 'foi', 'aos', 'nas', 'das', 'dos', 'pra', 'nos', 'tem', 'ele', 'ela', 'eles', 'elas', 'este', 'esse', 'isso', 'aquilo', 'muito', 'são', 'ser', 'está', 'quando', 'quem', 'qual', 'seu', 'sua', 'você', 'sobre', 'também', 'pelo', 'pela', 'nem', 'até', 'sem'}

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
                    palavras = re.findall(r'\b[a-záéíóúâêîôûãõç]{3,}\b', texto)
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
