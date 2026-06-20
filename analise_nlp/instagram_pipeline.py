from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple, NamedTuple
import re
from tqdm import tqdm

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Pandas é necessário para o pipeline de análise de comentários do Instagram. "
        "Instale com: pip install pandas"
    ) from exc

try:
    from apify_client import ApifyClient
except ImportError:  # pragma: no cover
    ApifyClient = None  # type: ignore

from analise_nlp.bert.inference import analyze_text as bert_analyze_text
from analise_nlp.rules.bad import check_bad
from analise_nlp.rules.spam import check_spam
from analise_nlp.rules.caps import check_caps
from analise_nlp.rules.hate import check_hate
from analise_nlp.rules.middle import check_middle
from analise_nlp.rules.nice import check_nice
from analise_nlp.rules.solida import check_solida


@dataclass(frozen=True)
class LexicoResult:
    is_hate: bool
    is_bad: bool
    is_nice: bool
    is_solida: bool
    is_caixa_alta: bool
    is_spam: bool
    is_middlelist: bool


class ModeloMlResult(NamedTuple):
    sentiment: str
    confidence: float


def modelo_ml(texto: str) -> ModeloMlResult:
    """Interface para o modelo de ML fine-tuned.
    
    Retorna sentimento ('Positivo', 'Negativo', 'Neutro') e confiança.
    """
    if not texto:
        return ModeloMlResult(sentiment='Neutro', confidence=0.0)

    try:
        raw_output = bert_analyze_text(texto)
    except Exception as exc:
        raise RuntimeError("Falha ao chamar o modelo de ML") from exc

    if not raw_output or not isinstance(raw_output, list):
        raise ValueError("Saída inesperada do modelo de ML: deve ser lista de previsões")

    prediction = raw_output[0]
    raw_label = str(prediction.get('label', '')).upper()
    raw_score = prediction.get('score', 0.0)

    confidence = float(raw_score) if isinstance(raw_score, (float, int)) else 0.0
    confidence = max(0.0, min(confidence, 1.0))

    # Mapeamento estrito para os novos padrões
    if raw_label in {'NEGATIVO', 'LABEL_0'}:
        sentiment = 'Negativo'
    elif raw_label in {'POSITIVO', 'LABEL_2'}:
        sentiment = 'Positivo'
    elif raw_label in {'NEUTRO', 'LABEL_1'}:
        sentiment = 'Neutro'
    else:
        sentiment = 'Neutro'

    return ModeloMlResult(sentiment=sentiment, confidence=confidence)


def analisador_lexico(texto: str) -> LexicoResult:
    """Interface para o analisador léxico."""
    texto_limpo = texto or ""

    is_hate = check_hate(texto_limpo)
    is_bad = check_bad(texto_limpo)
    is_nice = check_nice(texto_limpo)
    is_solida = check_solida(texto_limpo)
    is_middlelist = check_middle(texto_limpo)
    is_caixa_alta = check_caps(texto_limpo) or (
    texto_limpo.isupper() and any(char.isalpha() for char in texto_limpo)
    )

    is_spam = check_spam(texto_limpo)

    return LexicoResult(
        is_hate=is_hate,
        is_bad=is_bad,
        is_nice=is_nice,
        is_solida=is_solida,
        is_caixa_alta=is_caixa_alta,
        is_spam=is_spam,
        is_middlelist=is_middlelist,
    )


class ApifyInstagramScraper:
    def __init__(self, apify_token: str, max_posts: int = 10, max_comments_per_post: int = 200):
        if not apify_token:
            raise ValueError('O token APIFY_TOKEN não pode estar vazio.')
        if ApifyClient is None:
            raise ImportError(
                'Biblioteca apify-client não encontrada. Instale com: pip install apify-client'
            )

        self.apify_token = apify_token
        self.max_posts = max_posts
        self.max_comments_per_post = max_comments_per_post
        self._client = ApifyClient(token=apify_token)

    def _normalize_profile_handle(self, profile_handle: str) -> str:
        handle = (profile_handle or '').strip()
        if handle.startswith('@'):
            handle = handle[1:]
        if not handle:
            raise ValueError('Perfil Instagram inválido.')
        return handle

    def _run_actor(self, actor_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        actor = self._client.actor(actor_id)
        try:
            if hasattr(actor, 'call'):
                return actor.call(run_input=input_data)
            return actor.start(**input_data)
        except TypeError:
            return actor.call(body=input_data)

    def _list_dataset_items(self, dataset_id: str) -> List[Dict[str, Any]]:
        dataset = self._client.dataset(dataset_id)
        if hasattr(dataset, 'list_items'):
            result = dataset.list_items(limit=2000)
        else:
            result = dataset.get_items(limit=2000)

        if hasattr(result, 'items'):
            return result.items
        
        if isinstance(result, dict) and 'items' in result:
            return result['items']
        if isinstance(result, list):
            return result
        return []

    def _get_dataset_id(self, run_result: Dict[str, Any]) -> Optional[str]:
        return (
            run_result.get('defaultDatasetId')
            or run_result.get('defaultDataset')
            or run_result.get('datasetId')
            or run_result.get('dataset')
        )

    def extract_instagram_post_urls(self, profile_handle: str) -> List[str]:
        profile = self._normalize_profile_handle(profile_handle)
        profile_url = f"https://www.instagram.com/{profile}/"
        
        input_data = {
            'directUrls': [profile_url],
            'resultsType': 'posts',
            'resultsLimit': 10,
        }

        actor_id = 'apify/instagram-scraper' 
        print(f"\n[DEBUG] Chamando Actor: {actor_id} com payload: {input_data}")
        
        run_result = self._run_actor(actor_id, input_data)
        dataset_id = self._get_dataset_id(run_result)
        
        if not dataset_id:
            print("[DEBUG] ERRO: dataset_id não encontrado no run_result do perfil:", run_result)
            return []

        items = self._list_dataset_items(dataset_id)
        print(f"[DEBUG] Itens brutos retornados pelo dataset de perfil: {len(items)}")
        
        if items:
            print(f"[DEBUG] Chaves do primeiro item extraído: {list(items[0].keys())}")

        urls: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            
            candidate = (
                item.get('url')
                or item.get('postUrl')
                or item.get('link')
                or item.get('post_url')
                or item.get('shortcode_url')
                or item.get('displayUrl')
            )
            if candidate:
                urls.append(candidate)
                
        return list(dict.fromkeys(urls))

    def extract_instagram_comments(self, post_urls: List[str]) -> List[Dict[str, Any]]:
        if not post_urls:
            return []

        input_data = {
            'directUrls': post_urls, 
            'resultsType': 'comments', 
            'resultsLimit': self.max_comments_per_post,
          
        }
        
        actor_id = 'apify/instagram-scraper'
        print(f"\n[DEBUG] Chamando {actor_id} para COMENTÁRIOS com payload: {input_data}\n")
        
        run_result = self._run_actor(actor_id, input_data)
        dataset_id = self._get_dataset_id(run_result)
        
        if not dataset_id:
            raise RuntimeError('Não foi possível localizar o dataset do actor de comentários Instagram.')

        rows = self._list_dataset_items(dataset_id)
        
        comments: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
                
            if 'comments' in row and isinstance(row['comments'], list):
                comments.extend([c for c in row['comments'] if isinstance(c, dict)])
            elif 'latestComments' in row and isinstance(row['latestComments'], list):
                comments.extend([c for c in row['latestComments'] if isinstance(c, dict)])
            else:
                comments.append(row)
                
        print(f"[DEBUG] Total de comentários extraídos do dataset: {len(comments)}")
        return comments
    
    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(int(value), tz=timezone.utc)
            except OSError:
                return None
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None

        text = text.replace('Z', '+00:00')
        text = re.sub(r'/Date\((\d+)\)/', lambda m: datetime.fromtimestamp(int(m.group(1)) / 1000, tz=timezone.utc).isoformat(), text)

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass

        for fmt in (
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
        ):
            try:
                return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue

        return None

    def filter_comments_by_date(self, comments: Iterable[Dict[str, Any]], data_referencia: datetime) -> List[Dict[str, Any]]:
        if isinstance(data_referencia, date) and not isinstance(data_referencia, datetime):
            data_referencia = datetime.combine(data_referencia, datetime.min.time()).replace(tzinfo=timezone.utc)
        elif isinstance(data_referencia, datetime) and data_referencia.tzinfo is None:
            data_referencia = data_referencia.replace(tzinfo=timezone.utc)

        start_date = (data_referencia - timedelta(days=15)).date()
        end_date = (data_referencia + timedelta(days=15)).date()

        start = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)

        filtered: List[Dict[str, Any]] = []
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            raw_date = comment.get('publishedAt') or comment.get('createdAt') or comment.get('timestamp') or comment.get('date')
            published_at = self._parse_datetime(raw_date)
            if published_at is None:
                continue
            if start <= published_at <= end:
                filtered.append({**comment, 'publishedAt': published_at})
            else:
                print(f"[DEBUG-FILTRO] Barrado! Comentário feito em {published_at.strftime('%d/%m/%Y')} está fora da janela ({start_date.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')})")
        return filtered

    def scrape_comments_for_profile(self, profile_handle: str, data_referencia: Any) -> List[Dict[str, Any]]:
        print(f"\n--- INICIANDO PIPELINE PARA {profile_handle} ---")
        
        post_urls = self.extract_instagram_post_urls(profile_handle)
        print(f"[DEBUG] URLs de posts encontradas: {len(post_urls)}")

        comments = self.extract_instagram_comments(post_urls)
        print(f"[DEBUG] Total de comentários extraídos (antes do filtro): {len(comments)}")

        if comments:
            primeiro = comments[0]
            raw_date = primeiro.get('publishedAt') or primeiro.get('createdAt') or primeiro.get('timestamp') or primeiro.get('date')
            print(f"[DEBUG] Exemplo de data bruta vinda do Apify: '{raw_date}'")
            print(f"[DEBUG] Conversão da data: {self._parse_datetime(raw_date)}")

        filtered_comments = self.filter_comments_by_date(comments, data_referencia)
        print(f"[DEBUG] Comentários que passaram no filtro da data ({data_referencia}): {len(filtered_comments)}\n")

        return filtered_comments

    def scrape_comments_for_post(self, post_url: str, data_referencia: Any) -> List[Dict[str, Any]]:
        print(f"\n--- INICIANDO PIPELINE PARA POST ESPECÍFICO ---")
        print(f"[DEBUG] URL alvo: {post_url}")
        
        comments = self.extract_instagram_comments([post_url])
        print(f"[DEBUG] Total de comentários extraídos (antes do filtro): {len(comments)}")

        if comments:
            primeiro = comments[0]
            raw_date = primeiro.get('publishedAt') or primeiro.get('createdAt') or primeiro.get('timestamp') or primeiro.get('date')
            print(f"[DEBUG] Exemplo de data bruta vinda do Apify: '{raw_date}'")
            print(f"[DEBUG] Conversão da data: {self._parse_datetime(raw_date)}")

        filtered_comments = self.filter_comments_by_date(comments, data_referencia)
        print(f"[DEBUG] Comentários que passaram no filtro da data ({data_referencia}): {len(filtered_comments)}\n")

        return filtered_comments


class ScorerHibrido:
    def __init__(self, texto: str):
        self.texto = texto or ''
        self.lexico = analisador_lexico(self.texto)
        self.modelo = modelo_ml(self.texto)

    def _calcular_saldo_lexico(self) -> Tuple[int, str]:
        """Calcula o Saldo Base e aplica os Agregadores (Retorna padrão capitalizado)."""
        saldo = 0
        if self.lexico.is_hate: saldo -= 4
        if self.lexico.is_bad:  saldo -= 2
        if self.lexico.is_nice: saldo += 2
        if self.lexico.is_solida: saldo += 4

        agregadores = 0
        if self.lexico.is_caixa_alta: agregadores += 1
        if self.lexico.is_spam: agregadores += 1
        if self.lexico.is_middlelist: agregadores += 1

        if saldo > 0:
            s = 'Positivo'
            p = saldo + agregadores
        elif saldo < 0:
            s = 'Negativo'
            p = saldo - agregadores
        else:
            s = 'Neutro'
            p = 0
            
        return p, s

    def calcular_score_final(self) -> Tuple[str, float]:
        """Cruza os dados do léxico com o ML seguindo o diagrama de decisão."""
        # 1. Variáveis de entrada do Algoritmo (Léxico)
        pontos_lexico, sentimento_lexico = self._calcular_saldo_lexico()
        P = float(pontos_lexico) # Representa a variável 'P' no diagrama
        
        # 2. Variáveis de entrada do Modelo (ML)
        sentimento_ml = self.modelo.sentiment
        confianca_ml = self.modelo.confidence

        LIMITE_CONFIANCA = 0.80 # Limite geral para confiar na decisão do modelo
        LIMITE_CONFIANCA_POS = 0.75 # Mais tolerante para aceitar gírias positivas
        LIMITE_CONFIANCA_NEG = 0.98 # Mais rigoroso para classificar como sarcasmo/negativo
        
        # ---------------------------------------------------------
        # FLUXO 1: Algoritmo está NEUTRO (P == 0)
        # ---------------------------------------------------------
        if P == 0:
            if sentimento_ml in ['Positivo', 'Negativo'] and confianca_ml >= LIMITE_CONFIANCA:
                # Modelo tem certeza MAIOR que 80% -> Crava Positivo/Negativo
                sentimento_final = sentimento_ml
                # Atribui por padrão PESO 2, dependendo do retorno 'neg' ou 'pos'
                P_final = 2.0 if sentimento_ml == 'Positivo' else -2.0
            else:
                # Modelo tem certeza MENOR que 80% OU afirma que é Neutro -> Define Neutro
                sentimento_final = 'Neutro'
                P_final = 0.0
                
        # ---------------------------------------------------------
        # FLUXO 2: Algoritmo encontrou um sentimento (P != 0)
        # ---------------------------------------------------------
        else:
            # Define se concordam ou discordam
            concorda = (sentimento_lexico == 'Positivo' and sentimento_ml == 'Positivo') or \
                    (sentimento_lexico == 'Negativo' and sentimento_ml == 'Negativo')
            
            discorda = (sentimento_lexico == 'Positivo' and sentimento_ml == 'Negativo') or \
                    (sentimento_lexico == 'Negativo' and sentimento_ml == 'Positivo')
            
            if concorda:
                # Modelo CONCORDA em qualquer nível: mantém o peso de P
                sentimento_final = sentimento_lexico
                P_final = P
                
            elif discorda:
                if sentimento_ml == 'Positivo' and confianca_ml >= LIMITE_CONFIANCA_POS:
                    # ML acha que é positivo (Gíria) e tem confiança suficiente
                    sentimento_final = 'Positivo'
                    P_final = abs(P)
                    
                elif sentimento_ml == 'Negativo' and confianca_ml >= LIMITE_CONFIANCA_NEG:
                    # ML acha que é negativo (Sarcasmo) e tem ALTA confiança
                    sentimento_final = 'Negativo'
                    P_final = -abs(P)
                    
                else:
                    # Se não atingiu o limite de confiança para o seu respectivo lado
                    sentimento_final = 'Neutro'
                    P_final = 0.0
                    
            else:
                if confianca_ml >= LIMITE_CONFIANCA:
                    # Caso o modelo retorne Neutro (com qualquer confiança) e o Léxico tinha um valor
                    # O diagrama aponta a caixa azul tracejada: "... OU +80% que é NEUTRO" -> P = 0
                    sentimento_final = 'Neutro'
                    P_final = 0.0
                else:
                    # Caso o modelo retorne Neutro com MENOS de 80% de certeza, mantém o resultado do léxico
                    sentimento_final = sentimento_lexico
                    P_final = P

        return sentimento_final, P_final


def processar_comentarios(comentarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for comment in tqdm(comentarios, desc="Analisando comentários"):
        if not isinstance(comment, dict):
            continue

        texto = comment.get('text')
        if not texto:
            continue

        # INSTANCIA O SCORER E CHAMA O MÉTODO CORRETO
        scorer = ScorerHibrido(texto)
        sentimento_final, score_final = scorer.calcular_score_final()

        published_at = comment.get('data')
        #if isinstance(published_at, datetime):
        #    published_at_value = published_at
        #else:
        #    published_at_value = ApifyInstagramScraper._parse_datetime(published_at)

        # AGORA EXTRAIMOS DIRETO DO OBJETO SCORER
        rows.append(
            {
                'texto': texto,
                'account_name': comment.get('account_name'),
                'published_at': published_at,
                'is_hate': scorer.lexico.is_hate,
                'is_bad': scorer.lexico.is_bad,
                'is_nice': scorer.lexico.is_nice,
                'is_solida': scorer.lexico.is_solida,
                'is_caixa_alta': scorer.lexico.is_caixa_alta,
                'is_spam': scorer.lexico.is_spam,
                'is_middlelist': scorer.lexico.is_middlelist,
                'modelo_sentimento': scorer.modelo.sentiment,
                'modelo_confianca': scorer.modelo.confidence,
                'sentimento_base': sentimento_final,
                'score_p': score_final,
            }
        )

    return pd.DataFrame(rows)


def salvar_resultados_no_banco(df: pd.DataFrame, profile_handle: str = '') -> str:
    """Salva o DataFrame de resultados na tabela `api.Comment`."""
    if df is None or df.empty:
        print("[BANCO DE DADOS] Nenhum dado para salvar (DataFrame vazio).")
        return 'api_comment'

    try:
        from api.models import Comment
    except Exception as exc:
        print(f"[ERRO DE IMPORTAÇÃO] Não foi possível carregar o model Comment: {exc}")
        return 'api_comment'

    sucesso = 0
    erro = 0
    print(f"\n[BANCO DE DADOS] Iniciando salvamento de {len(df)} comentários da conta @{profile_handle}...")

    for _, row in df.iterrows():
        texto = row.get('texto') or ''
        account_name = row.get('account_name')
        score = row.get('score_p')
        try:
            score_val = float(score) if pd.notna(score) else 0.0
        except Exception:
            score_val = 0.0
            
        data_raw = row.get('published_at')
        data_val = None if pd.isna(data_raw) else data_raw
        
        hate_val = bool(row.get('is_hate', False))
        bad_val = bool(row.get('is_bad', False))
        nice_val = bool(row.get('is_nice', False))
        solida_val = bool(row.get('is_solida', False))
        caixa_alta_val = bool(row.get('is_caixa_alta', False))
        spam_val = bool(row.get('is_spam', False))
        middlelist_val = bool(row.get('is_middlelist', False))
        
        modelo_raw = row.get('modelo_sentimento')
        modelo_val = '' if pd.isna(modelo_raw) else str(modelo_raw)
            
        confianca_raw = row.get('modelo_confianca')
        confianca_val = float(confianca_raw) if pd.notna(confianca_raw) else 0.0

        try:
            Comment.objects.create(
                account_name=str(account_name),
                texto=str(texto),
                data=data_val,
                hate=hate_val,
                bad=bad_val,
                nice=nice_val,
                solida=solida_val,
                caixa_alta=caixa_alta_val,
                spam=spam_val,
                middlelist=middlelist_val,
                modelo=modelo_val,
                confianca=confianca_val,
                score_p=score_val
            )
            sucesso += 1
        except Exception as e:
            print(f"[ERRO AO SALVAR LINHA] {e} | Texto: {str(texto)[:30]}...")
            erro += 1
            
    return 'api_comment'


def run_instagram_sentiment_pipeline(
    profile_handle: str,
    apify_token: str,
    data_referencia: datetime,
    max_posts: int = 10,
    max_comments_per_post: int = 200,
) -> pd.DataFrame:
    """Executa todo o pipeline Apify + Scorer Híbrido e retorna um DataFrame."""
    scraper = ApifyInstagramScraper(
        apify_token=apify_token,
        max_posts=max_posts,
        max_comments_per_post=max_comments_per_post,
    )
    comments = scraper.scrape_comments_for_profile(profile_handle, data_referencia)
    return processar_comentarios(comments)