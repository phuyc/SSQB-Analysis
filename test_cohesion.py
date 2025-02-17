import spacy
import numpy as np
from nltk import download
from nltk.corpus import cmudict
import re

class TextAnalyzer:
    def __init__(self):
        """Initialize the analyzer with required NLP models and resources"""
        self.nlp = spacy.load('en_core_web_sm')
        self.pronounceable = cmudict.dict()  # For concreteness analysis
        
    def analyze_text(self, text):
        """
        Analyze text across all five major components.
        
        Args:
            text (str): The input text to analyze
            
        Returns:
            dict: Comprehensive analysis results
        """
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        results = {
            'narrativity': self._analyze_narrativity(doc),
            'syntactic_simplicity': self._analyze_syntax(doc),
            'word_concreteness': self._analyze_concreteness(doc),
            'referential_cohesion': self._analyze_referential_cohesion(doc),
            'deep_cohesion': self._analyze_deep_cohesion(doc)
        }
        
        return results
    
    def _analyze_narrativity(self, doc):
        """
        Analyze narrativity through multiple indicators:
        - Presence of characters/entities
        - Use of past tense verbs
        - Temporal expressions
        - Dialog indicators
        - Personal pronouns
        """
        total_words = len([token for token in doc if not token.is_space])
        
        # Character/entity analysis
        named_entities = len([ent for ent in doc.ents if ent.label_ in ['PERSON', 'ORG']])
        
        # Verb tense analysis
        past_tense_verbs = len([token for token in doc if token.tag_ in ['VBD', 'VBN']])
        all_verbs = len([token for token in doc if token.pos_ == 'VERB'])
        
        # Dialog indicators
        dialog_markers = len(re.findall(r'[""].*?[""]', doc.text))
        
        # Personal pronouns
        personal_pronouns = len([token for token in doc if token.tag_ == 'PRP'])
        
        # Temporal expressions
        temporal_markers = len([ent for ent in doc.ents if ent.label_ in ['TIME', 'DATE']])
        
        narrativity_score = {
            'character_density': named_entities / total_words,
            'past_tense_ratio': past_tense_verbs / all_verbs if all_verbs > 0 else 0,
            'dialog_density': dialog_markers / len(list(doc.sents)),
            'pronoun_density': personal_pronouns / total_words,
            'temporal_density': temporal_markers / total_words
        }
        
        # Calculate overall narrativity score
        weights = {'character_density': 0.3, 'past_tense_ratio': 0.3, 
                  'dialog_density': 0.15, 'pronoun_density': 0.15, 
                  'temporal_density': 0.1}
        
        narrativity_score['overall'] = sum(score * weights[metric] 
                                         for metric, score in narrativity_score.items() 
                                         if metric in weights)
        
        return narrativity_score
    
    def _analyze_syntax(self, doc):
        """
        Analyze syntactic simplicity through:
        - Words per sentence
        - Clauses per sentence
        - Words before main verb
        - Dependency tree depth
        - Passive voice usage
        """
        sentences = list(doc.sents)
        
        # Sentence length analysis
        words_per_sentence = [len([token for token in sent if not token.is_space]) 
                            for sent in sentences]
        
        # Clause analysis
        clauses_per_sentence = []
        for sent in sentences:
            clause_markers = len([token for token in sent 
                                if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
            clauses_per_sentence.append(clause_markers + 1)  # Add 1 for main clause
        
        # Words before main verb
        words_before_verb = []
        for sent in sentences:
            main_verb_index = next((i for i, token in enumerate(sent) 
                                  if token.pos_ == 'VERB'), len(sent))
            words_before_verb.append(main_verb_index)
        
        # Dependency tree depth
        def tree_depth(token):
            return 1 + max((tree_depth(child) for child in token.children), default=0)
        
        tree_depths = [tree_depth(sent.root) for sent in sentences]
        
        # Passive voice detection
        passive_constructs = len([token for token in doc 
                                if token.dep_ == 'nsubjpass'])
        
        syntax_scores = {
            'avg_words_per_sentence': np.mean(words_per_sentence),
            'avg_clauses': np.mean(clauses_per_sentence),
            'avg_words_before_verb': np.mean(words_before_verb),
            'avg_tree_depth': np.mean(tree_depths),
            'passive_ratio': passive_constructs / len(sentences)
        }
        
        # Convert to simplicity score (inverse of complexity)
        max_words = 20  # Benchmark for maximum "simple" sentence length
        max_clauses = 2  # Benchmark for maximum "simple" clause count
        max_depth = 5   # Benchmark for maximum "simple" tree depth
        
        syntax_scores['simplicity'] = 1 - np.mean([
            min(syntax_scores['avg_words_per_sentence'] / max_words, 1),
            min(syntax_scores['avg_clauses'] / max_clauses, 1),
            min(syntax_scores['avg_tree_depth'] / max_depth, 1),
            syntax_scores['passive_ratio']
        ])
        
        return syntax_scores
    
    def _analyze_concreteness(self, doc):
        """
        Analyze word concreteness through:
        - ImageNet word presence
        - Word length and syllable count
        - Abstract word patterns
        - Sensory words
        """
        def count_syllables(word):
            try:
                return len(self.pronounceable[word.lower()][0])
            except KeyError:
                return len(re.findall(r'[aeiou]+', word.lower()))
        
        # Lists of sensory and abstract words
        sensory_words = set(['see', 'hear', 'feel', 'smell', 'taste', 'touch', 
                           'look', 'sound', 'felt', 'sensed', 'observed'])
        abstract_suffixes = set(['tion', 'ness', 'ity', 'ance', 'ence', 'ism', 
                               'ment', 'ship', 'ability', 'ology'])
        
        total_content_words = len([token for token in doc 
                                 if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
        
        # Concrete word analysis
        concrete_indicators = {
            'sensory_words': len([token for token in doc 
                                if token.lemma_.lower() in sensory_words]),
            'abstract_derivatives': len([token for token in doc 
                                      if any(token.text.lower().endswith(suffix) 
                                           for suffix in abstract_suffixes)]),
            'avg_syllables': np.mean([count_syllables(token.text) 
                                    for token in doc if token.is_alpha]),
            'physical_objects': len([ent for ent in doc.ents 
                                   if ent.label_ in ['PRODUCT', 'OBJECT']])
        }
        
        # Calculate concreteness score
        concrete_indicators['concreteness_score'] = (
            (concrete_indicators['sensory_words'] + 
             concrete_indicators['physical_objects']) / total_content_words -
            concrete_indicators['abstract_derivatives'] / total_content_words
        )
        
        return concrete_indicators
    
    def _analyze_referential_cohesion(self, doc):
        """
        Analyze referential cohesion through:
        - Noun/pronoun reference chains
        - Word overlap between sentences
        - Semantic similarity between references
        """
        sentences = list(doc.sents)
        
        # Analyze reference chains
        references = {}
        for sent_idx, sent in enumerate(sentences):
            for token in sent:
                if token.pos_ in ['NOUN', 'PROPN', 'PRON']:
                    if token.text not in references:
                        references[token.text] = []
                    references[token.text].append(sent_idx)
        
        # Calculate reference metrics
        reference_lengths = [len(refs) for refs in references.values()]
        reference_gaps = [refs[i+1] - refs[i] 
                        for refs in references.values() 
                        for i in range(len(refs)-1)]
        
        # Word overlap analysis
        overlaps = []
        for i in range(len(sentences)-1):
            sent1_words = set([token.lemma_ for token in sentences[i] 
                             if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
            sent2_words = set([token.lemma_ for token in sentences[i+1] 
                             if token.pos_ in ['NOUN', 'VERB', 'ADJ']])
            if sent1_words and sent2_words:
                overlap = len(sent1_words & sent2_words) / len(sent1_words | sent2_words)
                overlaps.append(overlap)
        
        cohesion_scores = {
            'avg_reference_length': np.mean(reference_lengths) if reference_lengths else 0,
            'avg_reference_gap': np.mean(reference_gaps) if reference_gaps else 0,
            'word_overlap': np.mean(overlaps) if overlaps else 0
        }
        
        # Calculate overall referential cohesion score
        cohesion_scores['overall'] = (
            0.4 * cohesion_scores['avg_reference_length'] +
            0.3 * (1 - min(cohesion_scores['avg_reference_gap'] / 5, 1)) +
            0.3 * cohesion_scores['word_overlap']
        )
        
        return cohesion_scores
    
    def _analyze_deep_cohesion(self, doc):
        """
        Analyze deep cohesion through:
        - Causal/logical connections
        - Temporal progression
        - Semantic relationships
        - Topic consistency
        """
        sentences = list(doc.sents)
        
        # Analyze connectives
        causal_connectives = set(['because', 'therefore', 'thus', 'consequently', 
                                'hence', 'so'])
        logical_connectives = set(['however', 'although', 'unless', 'if', 
                                 'while', 'whereas'])
        temporal_connectives = set(['before', 'after', 'then', 'subsequently', 
                                  'finally', 'meanwhile'])
        
        # Count different types of connections
        connections = {
            'causal': len([token for token in doc 
                          if token.text.lower() in causal_connectives]),
            'logical': len([token for token in doc 
                          if token.text.lower() in logical_connectives]),
            'temporal': len([token for token in doc 
                           if token.text.lower() in temporal_connectives])
        }
        
        # Analyze semantic progression
        semantic_similarities = []
        for i in range(len(sentences)-1):
            sim = sentences[i].vector.dot(sentences[i+1].vector) / (
                np.linalg.norm(sentences[i].vector) * 
                np.linalg.norm(sentences[i+1].vector)
            )
            semantic_similarities.append(sim)
        
        # Topic consistency analysis
        doc_vector = doc.vector
        sentence_vectors = [sent.vector for sent in sentences]
        topic_similarities = [
            doc_vector.dot(sent_vector) / (
                np.linalg.norm(doc_vector) * np.linalg.norm(sent_vector)
            )
            for sent_vector in sentence_vectors
        ]
        
        cohesion_metrics = {
            'connection_density': sum(connections.values()) / len(sentences),
            'semantic_flow': np.mean(semantic_similarities) if semantic_similarities else 0,
            'topic_consistency': np.mean(topic_similarities),
            'connection_types': connections
        }
        
        # Calculate overall deep cohesion score
        cohesion_metrics['overall'] = (
            0.4 * cohesion_metrics['connection_density'] +
            0.3 * cohesion_metrics['semantic_flow'] +
            0.3 * cohesion_metrics['topic_consistency']
        )
        
        return cohesion_metrics

# Usage example
analyzer = TextAnalyzer()

text = """Researchers studying magnetosensation have determined why some soil-dwelling roundworms in the Southern Hemisphere move in the opposite direction of Earth’s magnetic field when searching for ______blank in the Northern Hemisphere, the magnetic field points down, into the ground, but in the Southern Hemisphere, it points up, toward the surface and away from worms’ food sources."""
analyzer = TextAnalyzer()
results = analyzer.analyze_text(text)

# Access individual components
print("Narrativity:", results['narrativity']['overall'])
print("Syntactic Simplicity:", results['syntactic_simplicity']['simplicity'])
print("Word Concreteness:", results['word_concreteness']['concreteness_score'])
print("Referential Cohesion:", results['referential_cohesion']['overall'])
print("Deep Cohesion:", results['deep_cohesion']['overall'])