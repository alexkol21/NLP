# NLP
#  NLP Paraphrasing Project with Transformers

Σε αυτή την εργασία χρησιμοποιούνται τεχνικές παραφραστικής ανακατασκευής (semantic paraphrasing) με μοντέλα NLP όπως το T5, το BART και το Pegasus. Σκοπός είναι να παραχθούν προτάσεις που είναι σημασιολογικά ισοδύναμες με κάποιες δεδομένες, καθώς και να μετρηθεί η συνάφεια τους χρησιμοποιώντας το cosine similarity και να γίνει οπτική αναπαράστασή τους με t-SNE.

---

##  Περιεχόμενα

- `t5_paraphrase.py`: Δημιουργεί παραφράσεις μέσω του μοντέλου T5
- `bart_paraphrase.py`: Δημιουργεί παραφράσεις με χρήση του μοντέλου BART
- `pegasus_paraphrase.py`: Δημιουργεί παραφράσεις με χρήση του μοντέλου Pegasus
- `evaluate_similarity.py`: Χρησιμοποιώντας το Sentence Transformers για να υπολογίσει τη συνάφεια cosine μεταξύ των αρχικών προτάσεων και των παραφρασμένων
- `visualize_embeddings.py`: Η τεχνική t-SNE χρησιμοποιείται για να απεικονίσει τις προτάσεις οπτικά
- `similarities.csv`: Αρχείο εξαγωγής με τις μετρήσεις συνάφειας
- `NLP_answers.docx`: Το αρχείο με τις απαντήσεις σε μορφή Word

---

##  Απαιτούμενα Πακέτα


```bash
pip install transformers sentence-transformers scikit-learn matplotlib pandas
