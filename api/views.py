from rest_framework import viewsets
from .models import Comment
from .serializers import CommentSerializer
from analise_nlp.preprocessing.clean_text import clean_text
from analise_nlp.rules.bad import check_bad
from analise_nlp.rules.middle import check_middle
from analise_nlp.rules.hate import check_hate
from analise_nlp.bert.inference import analyze_text
from analise_nlp.ensemble.scorer import ensemble_score

class CommentViewSet(viewsets.ModelViewSet):
    queryset = Comment.objects.all()
    serializer_class = CommentSerializer

    def perform_create(self, serializer):
        text = serializer.validated_data['text']
        cleaned = clean_text(text)
        rules_scores = {
            'bad': check_bad(cleaned),
            'middle': check_middle(cleaned),
            'hate': check_hate(cleaned),
        }
        bert_result = analyze_text(cleaned)
        score = ensemble_score(rules_scores, bert_result)
        serializer.save(score=score)