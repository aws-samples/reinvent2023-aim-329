"""
Copyright 2023 Amazon.com, Inc. or its affiliates.  All Rights Reserved.
SPDX-License-Identifier: MIT-0
"""
import boto3
from botocore.exceptions import ClientError
import ipywidgets as ipw
import json
import logging
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import os
import time
from tqdm import tqdm
from IPython.display import display, clear_output

# Create the logger
DEFAULT_LOG_LEVEL = logging.NOTSET
DEFAULT_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
log_level = os.environ.get('LOG_LEVEL')
match log_level:
    case '10':
        log_level = logging.DEBUG
    case '20':
        log_level = logging.INFO
    case '30':
        log_level = logging.WARNING
    case '40':
        log_level = logging.ERROR
    case '50':
        log_level = logging.CRITICAL
    case _:
        log_level = DEFAULT_LOG_LEVEL
log_format = os.environ.get('LOG_FORMAT')
if log_format is None:
    log_format = DEFAULT_LOG_FORMAT
elif len(log_format) == 0:
    log_format = DEFAULT_LOG_FORMAT
# Set the basic config for the lgger
logging.basicConfig(level=log_level, format=log_format)


# Function to create the OpenSearch client for AOSS
def auth_opensearch(host,  # serverless collection endpoint, without https://
                    region,
                    service='aoss'):
    # Get the credentials from the boto3 session
    logging.info("Getting session credentials...")
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
    logging.info("Completed getting session credentials.")

    # Create an OpenSearch client and use the request-signer
    logging.info("Creating the OpenSearch client...")
    os_client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        timeout=3000
    )
    logging.info("Completed creating the OpenSearch client.")
    return os_client


# Function to create an AOSS collection along with associated resources
def create_aoss_collection(aoss_client, collection_name, data_access_policy_name,
                           encryption_policy_name, network_policy_name, iam_role):
    # Delete the existing data access policy (if any) with the same name
    try:
        response = aoss_client.delete_access_policy(
            name=data_access_policy_name,
            type='data'
        )
        logging.debug(response)
        logging.warning('Data access policy deleted.')
        # Sleep for 1 second to wait for the delete request to process
        logging.info('Waiting for 1 second for the delete request to be processed.')
        time.sleep(1)
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Data access policy does not exist.')
        else:
            logging.error(error)

    # Create the data access policy
    data_access_policy = aoss_client.create_access_policy(
        description='Full access for specified IAM role.',
        name=data_access_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + collection_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + collection_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [iam_role],
                    'Description': 'Full access data policy'
                }
            ]
        ),
        type='data'
    )
    # Print the response
    logging.debug(data_access_policy)
    logging.info('Data access policy created.')

    # Delete the existing encryption policy (if any) with the same name
    try:
        response = aoss_client.delete_security_policy(
            name=encryption_policy_name,
            type='encryption'
        )
        logging.debug(response)
        logging.warning('Encryption policy deleted.')
        # Sleep for 1 second to wait for the delete request to process
        logging.info('Waiting for 1 second for the delete request to be processed.')
        time.sleep(1)
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Encryption policy does not exist.')
        else:
            logging.error(error)

    # Create the encryption policy
    encryption_policy = aoss_client.create_security_policy(
        description='Encrypt with AWS owned key.',
        name=encryption_policy_name,
        policy=json.dumps(
            {
                'Rules':
                    [
                        {
                            'Resource': ['collection/' + collection_name],
                            'ResourceType': 'collection'
                        }
                    ],
                'AWSOwnedKey': True
            }
        ),
        type='encryption'
    )
    # Print the response
    logging.debug(encryption_policy)
    logging.info('Encryption policy created.')

    # Delete the existing network policy (if any) with the same name
    try:
        response = aoss_client.delete_security_policy(
            name=network_policy_name,
            type='network'
        )
        logging.debug(response)
        logging.warning('Network policy deleted.')
        # Sleep for 1 second to wait for the delete request to process
        logging.info('Waiting for 1 second for the delete request to be processed.')
        time.sleep(1)
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Network policy does not exist.')
        else:
            logging.error(error)

    # Create the network policy
    network_policy = aoss_client.create_security_policy(
        description='Public network access.',
        name=network_policy_name,
        policy=json.dumps(
            [
                {
                    'Rules':
                        [
                            {
                                'Resource': ['collection/' + collection_name],
                                'ResourceType': 'collection'
                            }
                        ],
                    'AllowFromPublic': True
                }
            ]
        ),
        type='network'
    )
    # Print the response
    logging.debug(network_policy)
    logging.info('Network policy created.')

    # Delete the existing collection (if any) with the same name
    filtered_collections = aoss_client.list_collections(collectionFilters=
                                                        {'name': collection_name})['collectionSummaries']
    if len(filtered_collections) > 0:
        collection_id = filtered_collections[0]['id']
        response = aoss_client.delete_collection(
            id=collection_id
        )
        logging.debug(response)
        logging.warning('Collection deleted.')
        # Sleep for 5 seconds to wait for the delete request to process
        logging.info('Waiting for 5 seconds for the delete request to be processed.')
        time.sleep(5)
    else:
        logging.info('Collection does not exist.')

    # Create the collection
    logging.info('Starting collection creation...')
    collection = aoss_client.create_collection(
        description='Collection for Amazon Bedrock with RAG demo.',
        name=collection_name,
        type='VECTORSEARCH'
    )
    # Wait for the create request to process; keep checking status every 10 seconds
    while True:
        status = aoss_client.list_collections(collectionFilters=
                                              {'name': collection_name})['collectionSummaries'][0]['status']
        logging.info('Waiting for 10 seconds for the collection to be created...')
        if status in ('ACTIVE', 'FAILED'):
            break
        time.sleep(10)
    # Print the response
    logging.debug(collection)
    logging.info('Collection created.')
    # Return the collection detail
    return collection['createCollectionDetail']


# Function to delete an AOSS collection along with associated resources
def delete_aoss_collection(aoss_client, collection_id, data_access_policy_name,
                           encryption_policy_name, network_policy_name):
    # Delete the collection
    try:
        response = aoss_client.delete_collection(
            id=collection_id
        )
        logging.info(response)
        logging.info('Collection deleted.')
        # Sleep for 5 seconds to wait for the delete request to process
        logging.info('Waiting for 5 seconds for the delete request to be processed.')
        time.sleep(5)
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Collection does not exist.')
        else:
            logging.error(error)

    # Delete the existing data access policy (if any) with the same name
    try:
        response = aoss_client.delete_access_policy(
            name=data_access_policy_name,
            type='data'
        )
        logging.info(response)
        logging.info('Data access policy deleted.')
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Data access policy does not exist.')
        else:
            logging.error(error)

    # Delete the existing encryption policy (if any) with the same name
    try:
        response = aoss_client.delete_security_policy(
            name=encryption_policy_name,
            type='encryption'
        )
        logging.info(response)
        logging.info('Encryption policy deleted.')
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Encryption policy does not exist.')
        else:
            logging.error(error)

    # Delete the existing network policy (if any) with the same name
    try:
        response = aoss_client.delete_security_policy(
            name=network_policy_name,
            type='network'
        )
        logging.info(response)
        logging.info('Network policy deleted.')
    except ClientError as error:
        if error.response['Error']['Code'] == 'ResourceNotFoundException':
            logging.info('Network policy does not exist.')
        else:
            logging.error(error)


# Function to create the embeddings and prepare the documents to index
def prepare_index_document_list(br_embeddings, doc_type, doc_data_list, doc_link_list):
    # Loop through the chunked documents and create embeddings
    doc_list = []
    logging.info("Creating embeddings for {} document chunks and preparing documents to index...".format(doc_type))
    for i, doc_data in tqdm(enumerate(doc_data_list)):
        # Get the data formatted for indexing; time.sleep in place to avoid throttling
        doc_dict = {}
        doc_dict['content'] = doc_data
        doc_dict['title'] = doc_link_list[i].split('/')[-1].replace('.{}'.format(doc_type.lower()), '')
        doc_dict['source'] = doc_link_list[i]
        # Create the embedding for the document chunk
        embedding = br_embeddings.embed_query(text=doc_data)
        doc_dict['content-embedding'] = embedding
        # Store all the data in a list
        doc_list.append(doc_dict)
    logging.info("Completed creating embeddings for {} document chunks and preparing documents to index.".format(doc_type))
    logging.info("A total of {} documents have been prepared for indexing.".format(len(doc_list)))
    return doc_list


# Function to get counts from text
def get_counts_from_text(text):
    if text is None:
        text = ''
    char_count = len(text)
    word_count = len(text.split())
    return char_count, word_count


# Function to parse the prompt output
def parse_prompt_metadata(response):
    context = ''
    title = ''
    source = ''
    if 'source_documents' in response:
        source_documents = response['source_documents']
        if len(source_documents) > 0:
            source_metadata = source_documents[0].metadata
            context = source_metadata['content']
            title = source_metadata['title']
            source = source_metadata['source']
    return context, title, source


# Function to parse the prompt output for RetrievalQA
def parse_rqa_prompt_output(question, response):
    answer = ''
    if response is not None:
        if 'query' in response:
            question = response['query']
        if 'result' in response:
            answer = response['result']
        context, title, source = parse_prompt_metadata(response)
    return question, answer, context, title, source


# Function to parse the prompt output for ConversationalRetrievalChain
def parse_crc_prompt_output(response):
    chat_history = []
    question = ''
    answer = ''
    if response is not None:
        if 'chat_history' in response:
            chat_history = response['chat_history']
        if 'question' in response:
            question = response['question']
        if 'answer' in response:
            answer = response['answer']
        context, title, source = parse_prompt_metadata(response)
    return chat_history, question, answer, context, title, source


# Function to flatten the chat history array to text
def convert_crc_chat_history_to_text(response):
    chat_messages = []
    chat_history, question, answer, context, title, source = parse_crc_prompt_output(response)
    for chat_message in chat_history:
        chat_messages.append("\n")
        chat_message_type = chat_message.__class__.__name__
        match chat_message_type:
            case 'HumanMessage':
                chat_messages.append("Human: {}".format(chat_message.content))
            case 'AIMessage':
                chat_messages.append("Assistant: {}".format(chat_message.content))
    chat_messages.append("\n")
    return "\n".join(chat_messages)


# Function to create the model-specific inference parameters
def get_model_specific_inference_params(model_id, temperature, max_response_token_length):
    match model_id:
        case 'amazon.titan-tg1-large' | 'amazon.titan-text-lite-v1' | 'amazon.titan-text-express-v1':
            model_kwargs = {
                "temperature": temperature,
                "maxTokenCount": max_response_token_length
            }
        case 'anthropic.claude-instant-v1' | 'anthropic.claude-v1' | 'anthropic.claude-v2':
            model_kwargs = {
                "temperature": temperature,
                "max_tokens_to_sample": max_response_token_length
            }
        case 'ai21.j2-grande-instruct' | 'ai21.j2-jumbo-instruct' | 'ai21.j2-mid' | 'ai21.j2-mid-v1'\
             | 'ai21.j2-ultra' | 'ai21.j2-ultra-v1':
            model_kwargs = {
                "temperature": temperature,
                "maxTokens": max_response_token_length
            }
        case 'cohere.command-text-v14':
            model_kwargs = {
                "temperature": temperature,
                "max_tokens": max_response_token_length
            }
        case _:
            model_kwargs = None
    return model_kwargs


# Class that defines the ChatUX
class ChatUX:
    """
    A Chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain=False):
        self.qa = qa
        self.name = None
        self.b = None
        self.retrievalChain = retrievalChain
        self.out = ipw.Output()

    def start_chat(self):
        logging.info("Starting the chat assistant...")
        display(self.out)
        self.chat(None)
    def chat(self, _):
        if self.name is None:
            prompt = ""
        else:
            prompt = self.name.value
        if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
            logging.info("Thank you, that was a nice chat!")
            logging.info("Chat assistant ended.")
            return
        elif len(prompt) > 0:
            with self.out:
                thinking = ipw.Label(value="Thinking...")
                display(thinking)
                try:
                    if self.retrievalChain:
                        result = self.qa.invoke({'question': prompt})
                    else:
                        out = self.qa.invoke({'question': prompt,
                                              "chat_history": self.qa.memory.chat_memory.messages}) # , 'history':chat_history})
                        result = out['answer']
                        source = out['source_documents'][0].metadata['source']
                except Exception as e:
                    logging.error(e)
                    result = "No answer"
                thinking.value = ""
                display(f"AI: {result}")
                display(f"Here is my source: {source}")
                self.name.disabled = True
                self.b.disabled = True
                self.name = None

        if self.name is None:
            with self.out:
                self.name = ipw.Text(description="You:", placeholder='q to quit')
                self.b = ipw.Button(description="Send")
                self.b.on_click(self.chat)
                display(ipw.Box(children=(self.name, self.b)))