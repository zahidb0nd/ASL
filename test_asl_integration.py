import asl_utils
import sys

def test_asl_utils():
    print("Testing ASL Utils...")
    
    # Test Image URL
    url_a = asl_utils.get_asl_image_url('A')
    print(f"URL for A: {url_a}")
    assert url_a is not None, "Failed to get URL for A"
    
    # Test Description
    desc_a = asl_utils.get_asl_description('A')
    print(f"Description for A: {desc_a}")
    assert desc_a != "No description available.", "Failed to get description for A"
    
    # Test Word Suggestions
    sugs = asl_utils.get_word_suggestions('hel')
    print(f"Suggestions for 'hel': {sugs}")
    assert 'help' in sugs, "Failed to suggest 'help'"
    
    # Test Next Word suggestions (needs bigram model to be built)
    # The build_language_model runs on import, so verified if no error on import
    print("Bigram model loaded successfully.")
    
    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_asl_utils()
