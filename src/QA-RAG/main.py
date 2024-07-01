import requests
import json
import torch
import time
import nltk
import re
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

class LLM:
    """
    implements LLM related methods
    """

    def __init__(self, role="standard", response_format="string"):
        """
        class constructor
        """

        api_key = config('GROQ_API_KEY')
        self.client = Groq(api_key=api_key)
        self.role = role
        self.response_format = response_format
        if self.role == "JSON_Refiner":

            self.response_format = 'JSON'
            
            def refiner_function(response, prompt, json_refiner_example, max_retries = 5):
                """
                retries max_tries no. of times to get the desired JSON
                """

                retries = 0
                json_formatted_response = None

                refiner_prompt = f"""

                The prompt was:

                --- PROMPT ---

                {prompt}

                There was an error in the JSON formatting in your previous response below:

                --- RESPONSE ---

                {response}

                The correct JSON format is as follows:

                -- JSON Format ---
                {json_refiner_example}

                Try again and provide your response with the correct JSON format
                
                """

                while json_formatted_response is None:
                    retries += 1
                    if retries == max_retries:
                        print ("JSON formatting error")
                        exit()

                    try:
                        response = self.prompt_llm(refiner_prompt)
                        return json.loads(response)
                    except Exception as error:
                        print ("Exception occured: ",error)

                        #try to handle API timeout error
                        if 'Error code: 429' in str(error):
                            print ('~ 60 seconds, sleep timer progress ... ')
                            for i in tqdm(range(60)):
                                time.sleep(1)

            self.refiner_function = refiner_function

    def prompt_llm(self,prompt,max_retries=5):
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

        llm_response_string = None
        retries = 0
        while llm_response_string is None:
            retries += 1
            if retries == max_retries:
                print ("LLM error")
                exit()
            try:
                llm_response_string = str(chat_completion.choices[0].message.content)
                response = llm_response_string #shorthand
                if self.response_format == 'JSON':
                        json_object_in_response = '{'+response.split('{')[1].split('}')[0]+'}'
                        return json_object_in_response
                return llm_response_string

            except Exception as error:
                print ("Exception occured: ",error)

                #try to handle API timeout error
                if 'Error code: 429' in str(error):
                    print ('~ 60 seconds, sleep timer progress ... ')
                    for i in tqdm(range(60)):
                        time.sleep(1)

class Agents:

    @staticmethod
    def get_response_agent():
        """
        returns a standard llm instance
        """

        return LLM()

    @staticmethod
    def get_JSON_response_agent():
        """
        returns an LLM instance that response with JSON formatted string
        """

        return LLM(response_format='JSON')

    @staticmethod
    def JSON_output_refiner_agent():
        """
        returns 
        """
        
        return LLM(role='JSON_Refiner')

class Text_Preprocessor:

    @staticmethod
    def text_splitter(text, split_size=4):
        """
        splits text into splits of specified size
        """
        a, n = text, split_size
        k, m = divmod(len(a), n)
        return_list = list(a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
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
    def retrieve_context(text_splits,random_question,neural_net,top_k = 10):
        """
        retrieves top k context
        """

        query_vector = neural_net.vectorize(random_question); query_vectors = [query_vector for _ in range(len(text_splits))]
        split_vectors = [neural_net.vectorize(split) for split in text_splits]
        similarities = [neural_net.vector_similarity(x[0],x[1]).item() for x in zip(query_vectors,split_vectors)]
        top_3_idxs = [similarities.index(y) for y in sorted(similarities)[::-1][:top_k]]
        return '\n ===== \n'.join([text_splits[idx] for idx in top_3_idxs])

class QA_RAG:
    """
    implements methods for QA_Retr-based RAG
    """

    @staticmethod
    def run_demo(PERFORM_RESPONSE_CONTROLS = False):
        """
        executes random instance of QA_RAG on wiki article
        """

        random_articles = WikiFunctions.get_n_random_articles()
        
        for article in random_articles:
            article_title = str(article['title'])
            article_text = str(article['text'])

            quest_gen_agent = Agents.get_JSON_response_agent()
            quest_gen_prompt = f"""

                    Consider the following article information

                    Article Title: {article_title}
                    Article Text: {article_text}

                    Generate fact seeking questions from the given information
                    Ensure each question contains at least one proper noun.
                    Also make sure to provide your response in JSON Format as follows:

                    {{"Questions:" ["question1", "question2", ...., ]}}

            """

            questions = None

            if not PERFORM_RESPONSE_CONTROLS:
                questions = json.loads(quest_gen_agent.prompt_llm(quest_gen_prompt))

            if PERFORM_RESPONSE_CONTROLS:
                response = None
                json_refiner_prompt = f"""
                     Make sure to provide your response in JSON Format as follows:

                     {{"Questions:" ["question1", "question2", ...., ]}}
                """

                checker_agent = Agents.JSON_output_refiner_agent()
                
                try:
                    response = quest_gen_agent.prompt_llm(quest_gen_prompt)
                    questions = json.loads(response)
                except:
                    questions = checker_agent.refiner_function(response,quest_gen_prompt,json_refiner_prompt)

            query = choice(questions[list(questions.keys())[0]])
            print ('\n ----- QUERY ----- \n')
            print (query)

            qDecompAgent = Agents.get_JSON_response_agent()
            decomp_prompt = f"""

                    Consider the following query

                    ---- QUERY ----

                    {query}

                    Your task is to decompose the query into subqueries by making sure to follow the instructions below:

                    ---- INSTRUCTIONS ---
                    1. Ensure each subquery is fully self-contained and mentions at least one proper noun.
                    2. Make sure to only include information in the subquery that is explicitly mentioned in the original query.
                    3. If the query is already fully self-contained, respond with the original query.
                    4. Provide your response in JSON Format as follows:

                        {{"Decomposed Queries": ["query1","query2", ....,, ]}}

            """

            decompQs = None

            if not PERFORM_RESPONSE_CONTROLS:
                decompQs = json.loads(qDecompAgent.prompt_llm(decomp_prompt))

            if PERFORM_RESPONSE_CONTROLS:
                response = None
                json_refiner_prompt = f"""

                    Make sure to provide your response in JSON Format as follows:

                    {{"Decomposed Queries": ["query1","query2", ....,, ]}}
                """

                checker_agent = Agents.JSON_output_refiner_agent()

                try:
                    response = qDecompAgent.prompt_llm(decomp_prompt)
                    decompQs = json.loads(response)
                except:
                    decompQs = checker_agent.refiner_function(response,decomp_prompt,json_refiner_prompt)

            decompQs = decompQs[list(decompQs.keys())[0]]
            print ('\n ----- Decomposed Questions ----- \n')
            print (decompQs)

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

            gt_agent = Agents.get_JSON_response_agent()
            gt_agent_response, gt_response = None, None

            if not PERFORM_RESPONSE_CONTROLS:
                gt_agent_response = json.loads(gt_agent.prompt_llm(gt_prompt))
                gt_response = llm_response[list(gt_agent_response.keys())[0]]

            if PERFORM_RESPONSE_CONTROLS:
                gt_agent_response = None
                json_refiner_prompt = response_json_example
                refiner_agent = Agents.JSON_output_refiner_agent()
                try:
                    gt_agent_response = gt_agent.prompt_llm(gt_prompt)
                    gt_response = json.loads(gt_agent_response)
                except:
                    gt_response = refiner_agent.refiner_function(gt_agent_response,gt_prompt,json_refiner_prompt)

            gt_response = gt_response[list(gt_response.keys())[0]]
            print ('\n ------ GROUND TRUTH RESPONSE ------ \n')
            print (gt_response)

            text_splits = Text_Preprocessor.text_splitter(article_text)
            context = Retr.retrieve_context(text_splits,query,Neural_Net())

            QAagent = Agents.get_JSON_response_agent()
            qacomp_prompt = f"""

                    Consider the following text

                    ---- TEXT ----

                    {context}

                    Your task is to convert the context into a list of questions and answers by making sure to follow the instructions below:

                    ---- INSTRUCTIONS ---
                    1. Make sure to only include information that is explicitly mentioned in the context.
                    2. Provide your response in JSON Format as follows:

                        {{"Questions and Answers": ["question1, answer1", "question2, answer2", ...]}}

            """

            compQnAs = None

            if not PERFORM_RESPONSE_CONTROLS:
                compQnAs = json.loads(QAagent.prompt_llm(qacomp_prompt))

            if PERFORM_RESPONSE_CONTROLS:
                response = None
                json_refiner_prompt = f"""

                    Make sure to provide your response in JSON Format as follows:

                    {{"Questions and Answers": ["question1, answer1", "question2, answer2", ...]}}
                """

                checker_agent = Agents.JSON_output_refiner_agent()

                try:
                    response = QAagent.prompt_llm(qacomp_prompt)
                    compQnAs = json.loads(response)
                except:
                    compQnAs = checker_agent.refiner_function(response,qacomp_prompt,json_refiner_prompt)

            compQnAs = compQnAs[list(compQnAs.keys())[0]]
            print ('\n ----- CONTEXT ----- \n')
            print (compQnAs)

            responseAgent = Agents.get_JSON_response_agent()
            response_prompt = f"""

            Consider the following query and expanded query set. The expanded query set is provided to further clarify the query:

            --- QUERY ----

            {query}

            --- EXPANDED QUERY SET ---

            {str(decompQs)}:

            Consider also the following relevant questions and answers that might help respond to the query:

            --- RELEVANT QUESTIONS AND ANSWERS ----

            {str(compQnAs)}

            Your task is to respond to the query by making sure to follow the instructions below:

            ---- INSTRUCTIONS ---
            1. Respond only If the relevant questions and answers contain sufficient information to respond to the query.
            2. If the relevant questions and answers do not contain sufficient information, respond with "I do not know".
            3. Provide your response in JSON Format as follows:

            {{"Response": "your response"}}

            """

            response = None
            
            if not PERFORM_RESPONSE_CONTROLS:
                response = json.loads(responseAgent.prompt_llm(response_prompt))

            if PERFORM_RESPONSE_CONTROLS:
                intermediate_response = None
                json_refiner_prompt = f"""

                    Make sure to provide your response in JSON Format as follows:

                    {{"Response": "your response"}}
                """

                checker_agent = Agents.JSON_output_refiner_agent()

                try:
                    intermediate_response = responseAagent.prompt_llm(response_prompt)
                    response = json.loads(intermediate_response)
                except:
                    response = checker_agent.refiner_function(intermediate_response,response_prompt,json_refiner_prompt)

            response = response[list(response.keys())[0]]
            print ('\n ----- RESPONSE ----- \n')
            print (response)

            print ('\n ------ EVALUATION ------ \n')
            metrics = Evaluation.all_metrics(gt_response,response)        
            print (metrics) 

if __name__ == '__main__':
    QA_RAG.run_demo(PERFORM_RESPONSE_CONTROLS=True)