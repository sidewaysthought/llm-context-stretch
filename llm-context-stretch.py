import hashlib
import nltk
import requests
import spacy
from annoy import AnnoyIndex
from bs4 import BeautifulSoup
from rdflib import Graph, Namespace, Literal, URIRef
from sklearn.feature_extraction.text import CountVectorizer

# Download the NLTK punkt tokenizer if not already there
nltk.download('punkt')


class Llm_Context_Stretch:
    """
    A class for processing text into chunks of approximately max_tokens each.
    Chunks are split at sentence boundaries whenever possible.
    """


    def __init__(self):

        # Constants
        self.RDF_NAMESPACE = "db://llm-context-stretch#"
        self.TOKEN_LIMIT = 1000

        # Setup
        if not nltk.data.find('tokenizers/punkt'):
            nltk.download('punkt')

        # Create the RDF graph
        self.rdf_namespace = Namespace(self.RDF_NAMESPACE)
        self.rdf_database = Graph()
        self.next_vector_index = 0
        self.index_to_key = {}
        self.key_to_index = {} 

        # ... and the vector index
        self.vector_index = AnnoyIndex(300, 'angular')

        # Other useful stuff
        self.nlp = spacy.load('en_core_web_md')


    def generate_key(self, chunk) -> str:
        """
        Generate a key for the chunk based on its hash and keywords.
        
        Args:
            chunk: A chunk of text.
            
        Returns:
            A key for the chunk.
        """
        
        # Generate the hash
        hash_object = hashlib.md5(chunk.encode())
        hex_dig = hash_object.hexdigest()

        # Extract keywords
        vec = CountVectorizer().fit([chunk])
        keywords = vec.get_feature_names_out()[:3]  # take the first three as an example

        # Combine the hash and keywords to form the key
        key = f"{hex_dig}-{'-'.join(keywords)}"

        return key
    

    def make_embedding(self, chunk) -> list:
        """
        Embed the chunk using the spacy model.
        
        Args:
            chunk: A chunk of text.
            
        Returns:
            A list of word embeddings.
        """

        return self.nlp(chunk).vector
    

    def add_to_rdf(self, chunks = []) -> bool:
        """
        Add the chunks to the RDF graph.
        
        Args:
            chunks: A list of chunks of text.
            
        Returns:
            True if successful, False otherwise.
        """

        for i, chunk in enumerate(chunks):

            # Create a URIRef for the chunk based on its index
            print('Making key: ', i)
            key = self.generate_key(chunk)
            chunk_uri = URIRef(f"{self.RDF_NAMESPACE}/chunks/{key}")

            # Cross-references
            self.index_to_key[self.next_vector_index] = key
            self.key_to_index[key] = self.next_vector_index

            # Add the chunk to the graph
            print('Adding chunk: ', i)
            self.rdf_database.add((chunk_uri, self.rdf_namespace.text, Literal(chunk)))

            # Embed the chunk and add it to the index
            print('Adding embedding: ', i)
            embedding = self.make_embedding(chunk)
            self.vector_index.add_item(self.next_vector_index, embedding)
            self.next_vector_index += 1

        # Build the index
        self.vector_index.build(10)

        return True


    def chunk_text(self, text:str = '', max_tokens:int = 1000, overlap:int = 50) -> list:
        """
        Split text into chunks of approximately max_tokens each.
        Chunks are split at sentence boundaries whenever possible.

        Args:
            text: The text to split.
            max_tokens: The maximum number of tokens per chunk.
            overlap: The number of tokens to overlap between chunks.
        
        Returns:
            A list of chunks of text.
        """
        # Tokenize the text into sentences
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            # Tokenize the sentence into words
            tokens = nltk.word_tokenize(sentence)
            num_tokens = len(tokens)

            # If adding this sentence doesn't exceed the token limit, add it to the current chunk
            if current_tokens + num_tokens <= max_tokens:
                current_chunk.append(sentence)
                current_tokens += num_tokens
            else:
                # Otherwise, start a new chunk with this sentence
                # Include overlap from the end of the current chunk
                overlap_text = ' '.join(current_chunk[-overlap:])
                chunks.append(' '.join(current_chunk))
                current_chunk = [overlap_text, sentence]
                current_tokens = len(nltk.word_tokenize(' '.join(current_chunk)))

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


    def add_to_db(self, text:str = '') -> bool:
        """
        Split text into chunks of approximately max_tokens each.
        Chunks are split at sentence boundaries whenever possible.

        Args:
            text: The text to split.
            max_tokens: The maximum number of tokens per chunk.
        
        Returns:
            A list of chunks of text.
        """

        chunks = self.chunk_text(text, self.TOKEN_LIMIT)
        self.add_to_rdf(chunks)

        print(self.rdf_database.serialize(format='turtle'))

        return True

if __name__ == "__main__":
    
    processor = Llm_Context_Stretch()
    
    response = requests.get('https://en.wikipedia.org/wiki/Artificial_intelligence')
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paragraphs])
    chunks = processor.add_to_db(text)