import re
import json


class BPETokenizer:
    
    def __init__(self, encoder_file = "encoder.json", vocab_file = "vocab.bpe"):
        with open(encoder_file, 'r', encoding='utf-8') as f:
            self.encoder = json.load(f)
        self.decoder = {v:k for k,v in self.encoder.items()}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().split('\n')[1:-1]
        self.bpe_ranks = {tuple(line.split()): i for i, line in enumerate(vocab)}
        assert len(self.encoder) == 50257 and len(self.bpe_ranks) == 50000
        bs = list(range(33, 127)) + list(range(161, 256))
        xs = list(range(0, 33)) + list(range(127, 161))
        cs = bs[:] + [2**8 + i for i in range(len(xs))]
        self.byte_encoder = dict(zip(bs + xs, [chr(n) for n in cs]))
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}

    def encode(self, text, allowed_special=None):
        tokens = re.findall(r"""<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d| ?""" +
                            r"""\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""", text, re.UNICODE)
        def translate(token):
            if token == '<|endoftext|>':
                assert allowed_special and token in allowed_special
                return [token]
            word = tuple(''.join(self.byte_encoder[byte] for byte in token.encode('utf-8')))
            while len(word) != 1:
                pairs = set((word[i], word[i+1]) for i in range(len(word)-1))
                bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
                if bigram not in self.bpe_ranks:
                    break
                a, b = bigram
                new_word = []
                i = 0
                while i < len(word):
                    j = word.index(a, i) if a in word[i:] else len(word)
                    new_word.extend(word[i:j])
                    i = j
                    if i < len(word):
                        j = 2 if i < len(word)-1 and word[i] == a and word[i+1] == b else 1
                        new_word.append(a+b if j == 2 else word[i])
                        i += j
                word = tuple(new_word)
            return word
        return [self.encoder[_] for token in tokens for _ in translate(token)]

    def decode(self, tokens):
        tokens = [self.decoder[token] for token in tokens]
        buffer = bytearray([self.byte_decoder[c] for c in ''.join(tokens)])
        return buffer.decode('utf-8', errors='replace')
    
if __name__ == "__main__":
    # Initialize the tokenizer
    tokenizer = BPETokenizer()

    # Define a sample text to test encoding and decoding
    text1 = """Please output ONE sentence that has a subject, a verb, and possibly an object, the environment, and some details. Simply output the one-sentence caption only!!

        Correct examples:
        1. "A white sink under a mirror in a bathroom."
        2. "The two bears wondering about the point of the camera."
        3. "A very large bridge over lots of train tracks."

        Now, please provide the descriptive caption of the image provided at the beginning."""

    # Encode the text with allowed_special as a set
    encoded = tokenizer.encode(text1, allowed_special={"<|endoftext|>"})
    print("Encoded:", encoded)
    print(len(encoded)) # 132

    # Decode the encoded text
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    # Check if the decoded text matches the original text
    assert decoded == text1, "Decoded text does not match the original text"
    print("Test passed: Decoded text matches the original text.")
    
