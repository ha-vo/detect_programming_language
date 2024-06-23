from scipy.sparse import csr_matrix, lil_matrix
from glob import glob
import joblib
from bidict import bidict
# from StringIO import StringIO
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
import os
from collections import defaultdict, Counter
import re
import numpy as np
import pickle

LANGUAGE_ALIASES = {
    'java': 'java',
    'swing': 'java',
    'spring': 'java',
    'c': 'c',
    'c++': 'cpp',
    'cpp': 'cpp',
    'c#': 'csharp',
    'csharp': 'csharp',
    'c-sharp': 'csharp',
    'python': 'python',
    'python-2.7': 'python',
    'py': 'python',
    'django': 'python',
    'django/jinja': 'python',
    'numpy': 'python',
    'visualbasic': 'visualbasic',
    'visual basic': 'visualbasic',
    'visual-basic': 'visualbasic',
    'visual-basic-.net': 'visualbasic',
    'vb': 'visualbasic',
    'vb.net': 'visualbasic',
    'php': 'php',
    'javascript+php': 'php',
    'laravel': 'php',
    'symfony2': 'php',
    'javascript': 'javascript',
    'node.js': 'javascript',
    'jquery': 'javascript',
    'js': 'javascript',
    'angularjs': 'javascript',
    'perl': 'perl',
    'perl6': 'perl',
    'objc': 'objc',
    'objective c': 'objc',
    'objective-c': 'objc',
    'swift': 'swift',
    'go': 'go',
    'golang': 'go',
    'ruby': 'ruby',
    'rb': 'ruby',
    'ruby-on-rails': 'ruby',
    'ruby-on-rails-3': 'ruby',
    'matlab': 'matlab',
    'octave': 'matlab',
    'delphi': 'delphi',
    'groovy': 'groovy',
    'r': 'r',
    's': 'r',
    'sql': 'sql',
    'sqlite': 'sql',
    'sql-server': 'sql',
    'mssql': 'sql',
    'ms-sql': 'sql',
    'mysql': 'sql',
    'plsql': 'sql',
    'postgresql': 'sql',
    'scala': 'scala',
    'shell': 'shell',
    'bash': 'shell',
    'unix-shell': 'shell',
    'sh': 'shell',
    'lisp': 'lisp',
    'common lisp': 'lisp',
    'commonlisp': 'lisp',
    'scheme': 'lisp',
    'newlisp': 'lisp',
    'elisp': 'emacslisp',
    'emacs lisp': 'emacslisp',
    'emacs-lisp': 'emacslisp',
    'erlang': 'erlang',
    'rust': 'rust',
    'dart': 'dart',
    'f#': 'fsharp',
    'fsharp': 'fsharp',
    'f-sharp': 'fsharp',
    'clojure': 'clojure',
    'clj': 'clojure',
    'haskell': 'haskell',
    'json': 'json',
    'html': 'html',
    'xhtml': 'html',
    'html5': 'html',
    'xml': 'xml',
    'xml+django/jinja': 'xml',
    'xml+ruby': 'xml',
    'xml+php': 'xml',
    'css': 'css',
    'css+lasso': 'css',
    'css3': 'css',
    'latex': 'latex',
    'tex': 'latex',
    'lua': 'lua',
    'fortran': 'fortran',
    'prolog': 'prolog',
    'smalltalk': 'smalltalk',
    'ada': 'ada',
    'awk': 'awk',
}

LANGUAGES = frozenset(LANGUAGE_ALIASES.values())
DATA_FOLDER = 'test/data/'

def snippets_per_language(num_per_language=10000):
    ret = defaultdict(list)
    sources = os.listdir(DATA_FOLDER)
    for s in sources:
        source = os.path.join(DATA_FOLDER,s)
        for folder in os.listdir(source):
            if folder not in LANGUAGES:
                continue
            filenames = os.listdir(os.path.join(DATA_FOLDER,s, folder))
            if len(filenames) > num_per_language:
                filenames = np.random.choice(
                    filenames, num_per_language, replace=False)

            for i, filename in enumerate(filenames):             

                with open(os.path.join(DATA_FOLDER,s, folder, filename), encoding='latin-1') as f:
                    ret[folder].append(f.read())
    return ret

token_regex = re.compile(r'([a-zA-Z0-9_]+|[^ a-zA-Z0-9_\n\t]+)')

def tokenize_string(s):
    singles = token_regex.findall(s)
    pairs = ['%s %s' % (singles[i], singles[i + 1])
             for i in range(len(singles) - 1)]
    return singles + pairs

def get_token_frequencies(tokens):
    counter = Counter(tokens)
    num_tokens = float(len(tokens))
    return {t: n / num_tokens for t, n in counter.items()}

def tokenize_snippets(language_snippets, tokens_per_language=500):
    language_tokens = defaultdict(list)
    top_tokens_per_language = defaultdict(Counter)
    for language, snippets in language_snippets.items():
        for snippet in snippets:
            tokens = tokenize_string(snippet)
            frequencies = get_token_frequencies(tokens)
            language_tokens[language].append(frequencies)
            top_tokens_per_language[language].update(tokens)

    top_tokens = frozenset([
        t for language in language_tokens for t, _ in
        top_tokens_per_language[language].most_common(tokens_per_language)
    ])

    pruned_language_tokens = defaultdict(list)
    for language, token_frequencies in language_tokens.items():
        for frequencies in token_frequencies:
            pruned = {t: f for t, f in frequencies.items()
                      if t in top_tokens}
            pruned_language_tokens[language].append(pruned)

    return pruned_language_tokens

def create_dataset(language_tokens):
    all_tokens = sorted(set([t for _, snippets in language_tokens.items()
                             for snippet in snippets
                             for t, _ in snippet.items()]))

    
    token_index = bidict({t: i for i, t in enumerate(all_tokens)})

    language_index = bidict({lang: i for i, lang in enumerate(sorted(language_tokens))})

    num_snippets = np.sum([len(s) for s in language_tokens.values()])
    features = lil_matrix((num_snippets, len(all_tokens)), dtype=np.float32)
    labels = np.zeros(num_snippets, dtype=np.float32)

    i = 0
    for language, snippets in language_tokens.items():
        for snippet in snippets:
            for token, frequency in snippet.items():
                features[i, token_index[token]] = frequency
            labels[i] = language_index[language]
            i += 1

            

    return (features.tocsr(), labels, token_index, language_index)

data_path = "test/data/rosetta/c/99-bottles-of-beer-5.c"
s = open(data_path, 'r').read()
tokens = tokenize_string(s)
frequencies = get_token_frequencies(tokens)
snippets = snippets_per_language()
language_tokens = tokenize_snippets(snippets)
features, labels, token_index, language_index = create_dataset(language_tokens)
test_features = []
test_feature = np.zeros(features.shape[1])
for token, freq in frequencies.items():
    if token in token_index:
        test_feature[token_index[token]] = freq
test_features.append(test_feature)
test_features = csr_matrix(test_features)
with open('assets/classifier.pkl', 'rb') as f:
    clf = pickle.load(f)
preds = clf.predict(test_features)
probas = clf.predict_proba(test_features)
langs = [language_index.inv[i] for i in range(len(language_index))]
lang_probas = max([sorted(zip(p, langs)) for p in probas])
print(lang_probas[-1][1])


   
