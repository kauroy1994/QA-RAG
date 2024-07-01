import requests
import json
import torch
import time
import nltk
import torch.nn as nn
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
from bs4 import BeautifulSoup
from tqdm import tqdm
from decouple import config
from groq import Groq
from random import choice
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import SmoothingFunction
from rouge_score import rouge_scorer

nltk.download('punkt')
nltk.download('wordnet')

class Evaluation(object):

    @staticmethod
    def all_metrics(s1, s2):
        s1_tokens = word_tokenize(s1.strip())
        s2_tokens = word_tokenize(s2.strip())
        cc = SmoothingFunction()
        BLEUscore1 = nltk.translate.bleu_score.sentence_bleu([s1_tokens], s2_tokens, smoothing_function=cc.method5, weights=[(1.0)])
        BLEUscore4 = nltk.translate.bleu_score.sentence_bleu([s1_tokens], s2_tokens, smoothing_function=cc.method5, weights=[(0.25,0.25,0.25,0.25)])
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        scores = scorer.score(s1,s2)
        meteor_score = nltk.translate.meteor_score.meteor_score([s1_tokens], s2_tokens)

        return {"Rouge_scores": [str(scores)],
        "BlEU-1":[str(BLEUscore1)],
        "BlEU-4":[str(BLEUscore4)],
        "METEOR":[str(meteor_score)]}

class WikiFunctions(object):
    """
    implements wikipedia based data loading methods
    """

    @staticmethod
    def get_random_article():
        """
        gets a random passage from wikipedia
        """
        try:
            response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
            soup = BeautifulSoup(response.content, "html.parser")
            title = soup.find(class_="firstHeading").text
            text = soup.get_text()
            return title, text
        except RuntimeError as error:
            print ("\n -- Error Occured --- \n",error)
            exit()

    @staticmethod
    def get_n_random_articles(n=1):

        try:
            articles = []
            for _ in range(n):
                title, text = WikiFunctions.get_random_article()
                articles.append({
                "title": title,
                "text": text,
            })

            return articles

        except RuntimeError as error:
            print ("\n -- Error Occured --- \n",error)
            exit()

class LLM(object):
    """
    implements LLM related methods
    """

    def __init__(self):
        """
        class constructor
        """

        api_key = config('GROQ_API_KEY')
        self.client = Groq(api_key=api_key)

    def prompt_llm(self,prompt):
        """
        returns llm response based on prompt
        """

        client = self.client

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    }
                    ],
                    temperature = 0.0,
                    model="mixtral-8x7b-32768",
                    )

        llm_response_string = str(chat_completion.choices[0].message.content)
        return llm_response_string

    def response_capture(self,response, format='JSON'):
        """
        returns common aspects of 'formatted' object in response and response set (if provided),
        (else) returns 'formatted' object
        """
        if format == 'JSON':
            json_object_in_response = '{'+response.split('{')[1].split('}')[0]+'}'
            return json_object_in_response

    def json_prompt_llm(self,prompt,json_example = None,max_retries=5):
        """
        prompts the llm and tries 5 times to format output as a valid JSON
        """

        questions = None
        retries = 0
        response = None    

        while True:

            retries += 1

            if not questions is None or retries == max_retries:
                return questions

            try:
                response = self.response_capture(self.prompt_llm(prompt))
                questions = json.loads(response)

            except Exception as e:
                print (e)
                if 'Error code: 429' in str(e):
                    print ('~ 60 seconds, sleep timer progress ... ')
                    for i in tqdm(range(60)):
                        time.sleep(1)

                new_prompt = f"""

                The prompt was:

                --- PROMPT ---

                {prompt}

                There was an error in the JSON formatting in your previous response below:

                --- RESPONSE ---

                {response}

                The correct JSON format is as follows:

                -- JSON Format ---
                {json_example}

                Try again and provide your response with the correct JSON format
                
                """

                response = self.response_capture(self.prompt_llm(new_prompt))
                questions = json.loads(response)

class Text_Preprocessor:
    
    @staticmethod
    def text_splitter(text):
        """
        splits text into splits of specified size
        """

        return_list = sent_tokenize(text)
        processed_return_list = []
        for item in return_list:
            processed_return_list.append(';'.join([sub_item for sub_item in item.split('\n') if item.strip()]))

        return processed_return_list

class Neural_Net:

    def __init__(self):

        def vector_similarity(vector1,vector2):
            """
            implements a vector similarity instance
            """

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            return cos(vector1,vector2)

        def vectorize(sentence):
            """
            implements a vectorizer instance
            """

            embedding_model = TransformerDocumentEmbeddings('bert-base-uncased')
            sentence = Sentence(sentence)
            embedding_model.embed(sentence)
            return sentence.embedding

        self.vectorize = vectorize
        self.vector_similarity = vector_similarity

class Retr:

    @staticmethod
    def retrieve_context(text_splits,random_question,neural_net,top_k = 3):
        """
        retrieves top k context
        """

        query_vector = neural_net.vectorize(random_question); query_vectors = [query_vector for _ in range(len(text_splits))]
        split_vectors = [neural_net.vectorize(split) for split in text_splits]
        similarities = [neural_net.vector_similarity(x[0],x[1]).item() for x in zip(query_vectors,split_vectors)]
        top_3_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return '\n ===== \n'.join([text_splits[idx] for idx in top_3_idxs])

class Sentence_Window_RAG:

    """
    implements methods for query rewriting and fusion-based RAG
    """
    
    @staticmethod
    def run_demo():
        """
        executes a random instance of QR_Fusion RAG on random wiki articles
        """

        random_articles = WikiFunctions.get_n_random_articles()
        random_questions = []

        for article in random_articles:
            article_title = str(article['title'])
            article_text = str(article['text'])

            text_splits = Text_Preprocessor.text_splitter(article_text)

            llm = LLM()
            prompt = f"""

                    Consider the following article information

                    Article Title: {article_title}
                    Article Text: {article_text}

                    Generate fact seeking questions from the given information
                    Ensure each question contains at least one proper noun.
                    Also make sure to provide your response in JSON Format as follows:

                    {{"Questions:" ["question1", "question2", ...., ]}}

            """

            json_example = f"""
                    Make sure to provide your response in JSON Format as follows:

                    {{"Questions:" ["question1", "question2", ...., ]}}
            """

            questions = llm.json_prompt_llm(prompt,json_example=json_example)
            query = choice(questions[list(questions.keys())[0]])

            print ('\n ------ QUERY ------ \n')
            print (query)

            gt_prompt = f"""

                Consider the query, passge information given below:

                ----- QUERY -----

                {query}

                ----- PASSAGE -----

                Title: {article_title}
                Text: {article_text}

                Respond with the answer to the query
                Ensure your responds is to the point and JSON formatted as shown below:

                {{"Response": "your response"}}

            """

            response_json_example = f"""
                    Make sure to provide your response in JSON Format as follows:

                    {{"Response": "your response"}}
            """

            llm_response = llm.json_prompt_llm(gt_prompt,json_example=response_json_example)
            gt_response = llm_response[list(llm_response.keys())[0]]

            print ('\n ------ GROUND TRUTH RESPONSE ------ \n')
            print (gt_response)

            context = Retr.retrieve_context(text_splits,query,Neural_Net())

            print ('\n ------ CONTEXT ------ \n')
            print (context)

            response_prompt = f"""

            Consider the following query:

            --- QUERY ----

            {query}

            Consider also the following context:

            --- CONTEXT ----

            {context}

            If the context seems relevant to the query, provide a response. If the context does not seem relevant to the query, respond with "I do not know".
            Ensure your response is to the point, and JSON formatted as follows:

            {{"Response": "response"}}

            """

            json_example = f"""
                    Make sure to provide your response in JSON Format as follows:

                    {{"Response": "rewritten query"}}
            """

            llm_response = llm.json_prompt_llm(response_prompt,json_example=json_example)
            response_value = llm_response[list(llm_response.keys())[0]]

            print ('\n ------ RESPONSE ------ \n')
            print (response_value)

            print ('\n ------ EVALUATION ------ \n')
            metrics = Evaluation.all_metrics(gt_response,response_value)        
            print (metrics)    

if __name__ == '__main__':
    Sentence_Window_RAG.run_demo()

