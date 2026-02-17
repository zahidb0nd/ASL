import nltk
from nltk.corpus import words, brown
from nltk import bigrams
from collections import defaultdict, Counter

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/words')
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('words')
    nltk.download('brown')

# Build language model (moved from app.py to here for modularity)
def build_language_model():
    english_words = set(w.lower() for w in words.words() if len(w) > 2)
    brown_words = [w.lower() for w in brown.words()]
    
    # Frequency distribution
    word_freq = Counter(brown_words)
    
    bigram_model = defaultdict(Counter)
    
    for w1, w2 in bigrams(brown_words):
        if w1.isalpha() and w2.isalpha():
            bigram_model[w1][w2] += 1
    
    return english_words, bigram_model, word_freq

# Initialize model once
ENGLISH_WORDS, BIGRAM_MODEL, WORD_FREQ = build_language_model()

# Wikimedia Commons Public Domain / CC BY-SA 4.0 images
# Using Gallaudet font or similar clear hand signs
ASL_IMAGES = {
    'A': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Sign_language_A.svg/1200px-Sign_language_A.svg.png',
    'B': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Sign_language_B.svg/1200px-Sign_language_B.svg.png',
    'C': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Sign_language_C.svg/1200px-Sign_language_C.svg.png',
    'D': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Sign_language_D.svg/1200px-Sign_language_D.svg.png',
    'E': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Sign_language_E.svg/1200px-Sign_language_E.svg.png',
    'F': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/Sign_language_F.svg/1200px-Sign_language_F.svg.png',
    'G': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Sign_language_G.svg/1200px-Sign_language_G.svg.png',
    'H': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Sign_language_H.svg/1200px-Sign_language_H.svg.png',
    'I': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7d/Sign_language_I.svg/1200px-Sign_language_I.svg.png',
    'J': 'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Sign_language_J.svg/1200px-Sign_language_J.svg.png',
    'K': 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Sign_language_K.svg/1200px-Sign_language_K.svg.png',
    'L': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Sign_language_L.svg/1200px-Sign_language_L.svg.png',
    'M': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c4/Sign_language_M.svg/1200px-Sign_language_M.svg.png',
    'N': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Sign_language_N.svg/1200px-Sign_language_N.svg.png',
    'O': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Sign_language_O.svg/1200px-Sign_language_O.svg.png',
    'P': 'https://upload.wikimedia.org/wikipedia/commons/thumb/0/08/Sign_language_P.svg/1200px-Sign_language_P.svg.png',
    'Q': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Sign_language_Q.svg/1200px-Sign_language_Q.svg.png',
    'R': 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Sign_language_R.svg/1200px-Sign_language_R.svg.png',
    'S': 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Sign_language_S.svg/1200px-Sign_language_S.svg.png',
    'T': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Sign_language_T.svg/1200px-Sign_language_T.svg.png',
    'U': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Sign_language_U.svg/1200px-Sign_language_U.svg.png',
    'V': 'https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Sign_language_V.svg/1200px-Sign_language_V.svg.png',
    'W': 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/86/Sign_language_W.svg/1200px-Sign_language_W.svg.png',
    'X': 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Sign_language_X.svg/1200px-Sign_language_X.svg.png',
    'Y': 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Sign_language_Y.svg/1200px-Sign_language_Y.svg.png',
    'Z': 'https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Sign_language_Z.svg/1200px-Sign_language_Z.svg.png'
}

def get_asl_image_url(letter):
    """Returns the URL of the ASL sign image for the given letter."""
    return ASL_IMAGES.get(letter.upper(), None)

def get_word_suggestions(partial_word, max_suggestions=5):
    """Get word suggestions based on partial input, sorted by frequency."""
    if not partial_word or len(partial_word) < 2:
        return []
    
    partial_lower = partial_word.lower()
    suggestions = [w for w in ENGLISH_WORDS if w.startswith(partial_lower)]
    
    # Sort by frequency (descending) then length (ascending) then alphabetical
    suggestions.sort(key=lambda x: (-WORD_FREQ.get(x, 0), len(x), x))
    
    return suggestions[:max_suggestions]

def get_next_word_suggestions(last_word, max_suggestions=3):
    """Get next word suggestions based on bigram model."""
    if not last_word:
        return []
    
    last_word_lower = last_word.lower()
    
    if last_word_lower in BIGRAM_MODEL:
        next_words = BIGRAM_MODEL[last_word_lower].most_common(max_suggestions)
        return [word for word, _ in next_words]
    
    return []

def get_asl_description(letter):
    """Returns a text description of the hand shape for a letter."""
    descriptions = {
        'A': 'Fist with thumb against the side of the index finger.',
        'B': 'Flat hand with thumb tucked in front of palm.',
        'C': 'Hand curved like a C.',
        'D': 'Index finger up, thumb and other fingers forming a circle.',
        'E': 'Fingers curled down, thumb tucked underneath.',
        'F': 'Index and thumb forming a circle, other three fingers up.',
        'G': 'Index finger pointing left, thumb parallel.',
        'H': 'Index and middle fingers pointing left, thumb parallel.',
        'I': 'Pinky finger up.',
        'J': 'Pinky finger traces a J.',
        'K': 'Index finger up, middle finger forward, thumb between.',
        'L': 'Thumb and index finger form an L.',
        'M': 'Three fingers over thumb.',
        'N': 'Two fingers over thumb.',
        'O': 'Fingertips meet thumb tip to form an O.',
        'P': 'Index finger forward, middle finger down, thumb between.',
        'Q': 'Index and thumb pointing down.',
        'R': 'Index and middle fingers crossed.',
        'S': 'Fist with thumb over fingers.',
        'T': 'Thumb between index and middle finger.',
        'U': 'Index and middle fingers up and together.',
        'V': 'Index and middle fingers up and spread (V shape).',
        'W': 'Three fingers up and spread.',
        'X': 'Index finger crooked.',
        'Y': 'Thumb and pinky out, others curled.',
        'Z': 'Index finger traces a Z in the air.'
    }
    return descriptions.get(letter.upper(), "No description available.")
