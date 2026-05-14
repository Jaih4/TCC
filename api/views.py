from rest_framework import viewsets
from .models import Comment
from .serializers import CommentSerializer
from analise_nlp.preprocessing.clean_text import clean_text
from analise_nlp.rules.bad import check_profanity
from analise_nlp.rules.middle import check_threats
from analise_nlp.rules.hate import check_discrimination
from analise_nlp.bert.inference import analyze_text
from analise_nlp.ensemble.scorer import ensemble_score

class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

    def perform_create(self, serializer):
        text = serializer.validated_data['text']
        cleaned = clean_text(text)
        rules_scores = {
            'profanity': check_profanity(cleaned),
            'threats': check_threats(cleaned),
            'discrimination': check_discrimination(cleaned),
        }
        bert_result = analyze_text(cleaned)
        score = ensemble_score(rules_scores, bert_result)
        serializer.save(score=score)