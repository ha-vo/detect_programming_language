from scipy.sparse import csr_matrix, lil_matrix
from glob import glob
from bidict import bidict
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import os
from collections import defaultdict, Counter
import re
import numpy as np
import pickle

# TODO:
#   * download stackoverflow dump from archive.org
#   * use code samples in questions as ground truth if one
#     and only one language is in the tags
#   * https://archive.org/details/stackexchange

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
                if i % 1000 == 0:
                    print (folder, i)

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

    print (len(all_tokens))

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

            if i % 10000 == 0:
                print ('%.2f%%' % (100 * (i + 1) / float(num_snippets)))

    return (features.tocsr(), labels, token_index, language_index)

def get_classifier(n_jobs=1):
    # parameters found with random grid search
    return RandomForestClassifier(
        n_estimators=11,
        criterion='gini',
        min_samples_split=2,
        max_depth=100,
        min_samples_leaf=5,
        max_leaf_nodes=None,
        n_jobs=n_jobs,
    )

def train_classifier(features, labels):
    clf = get_classifier(n_jobs=4)
    score = np.mean(cross_val_score(clf, features, labels, n_jobs=4))
    return clf.fit(features, labels), score

def test_test_set_fast(clf, test_features=None, test_labels=None, codes=None):
    if test_features is None:
        test_features, test_labels, codes = get_test_set()

    preds = clf.predict(test_features)
    probas = clf.predict_proba(test_features)
    langs = [language_index.inv[i] for i in range(len(language_index))]
    lang_probas = [sorted(zip(p, langs)) for p in probas]
    return np.mean(preds == test_labels), preds

def get_test_set():
    test_features = []
    test_labels = []
    codes = []
    root_path = 'test/data/stackoverflow'
    labels = os.listdir(root_path)
    for d in labels:
        dir_path =  root_path + '/' + d     
        filenames = os.listdir(dir_path)
        for f in filenames:
            if os.path.basename(f) == 'filenames':
                continue
            language = d
            filename = dir_path + '/' + f
            try:
                with open(filename, 'rb') as file:
                    s = file.read().decode('latin-1')
            except UnicodeDecodeError as e:
                print(f"UnicodeDecodeError: {e}")
                return
            codes.append(s)
            tokens = tokenize_string(s)
            frequencies = get_token_frequencies(tokens)
            test_feature = np.zeros(features.shape[1])
            for token, freq in frequencies.items():
                if token in token_index:
                    test_feature[token_index[token]] = freq
            test_features.append(test_feature)
            test_labels.append(language_index[language])
    return csr_matrix(test_features), np.array(test_labels), codes

def compare_pygments():
    from pygments.lexers import guess_lexer, ClassNotFound

    total = 0
    correct = 0
    bad = defaultdict(int)

    for lang in glob('test/data/linguist/*'):
        for filename in glob('%s/*' % lang):
            if os.path.basename(filename) == 'filenames':
                continue
            with open(filename) as f:
                try:
                    pred = guess_lexer(f.read()).name.lower()
                    if pred in LANGUAGE_ALIASES:
                        pred = LANGUAGE_ALIASES[pred]
                    else:
                        bad[pred] += 1
                    if pred == lang.split('/')[-1]:
                        correct += 1
                except ClassNotFound:
                    pred = 'unknown'
            total += 1

    return correct / float(total), bad
def write_classifier(clf):
    with open('assets/classifier.pkl', 'wb') as file:
        pickle.dump(clf, file)


snippets = snippets_per_language()
language_tokens = tokenize_snippets(snippets)
print(language_tokens)
features, labels, token_index, language_index = create_dataset(language_tokens)
clf, score = train_classifier(features, labels)
write_classifier(clf)
print(test_test_set_fast(clf))


