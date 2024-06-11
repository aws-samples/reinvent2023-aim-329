## Build a chat assistant with Amazon Bedrock

This repository contains the code samples rthat will let participants explore how to use the [Retrieval Augmented Generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html) architecture with [Amazon Bedrock](https://aws.amazon.com/bedrock/) and [Amazon OpenSearch Serverless (AOSS)](https://aws.amazon.com/opensearch-service/features/serverless/) to quickly build a secure chat assistant that uses the most up-to-date information to converse with users. Participants will also learn how this chat assistant will use dialog-guided information retrieval to respond to users.

### Overview

[Amazon Bedrock](https://aws.amazon.com/bedrock/) is a fully managed service that offers a choice of high-performing Foundation Models (FMs) from leading AI companies accessible through a single API, along with a broad set of capabilities you need to build generative AI applications, simplifying development while maintaining privacy and security.

[Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model) are a type of Foundation Model that can take natural langauge as input, with the ability to process and understand it, and produce natural language as the output. LLMs can also can perform tasks like classification, summarization, simplification, entity recognition, etc.

LLMs are usually trained offline with data that is available until that point of time. As a result, LLMs will not have knowledge of the world after that date. Additionally, LLMs are trained on very general domain corpora, making them less effective for domain-specific tasks. And then, LLMs have the tendency to hallucinate where the model generates text that is incorrect, nonsensical, or not real. Using a [Retrieval Augment Generation (RAG)](https://docs.aws.amazon.com/sagemaker/latest/dg/jumpstart-foundation-models-customize-rag.html) mechanism can help mitigate all these issues. A RAG architecture involves retrieving data that closely matches the text in the user's prompt, from an external datasource, and using it to augment the prompt before sending to the LLM. This prompt augmentation will provide the context that the LLM can use to respond to the prompt.

In this session, we will use [Amazon OpenSearch Serverless (AOSS)](https://aws.amazon.com/opensearch-service/features/serverless/) as the external datasource to store and search the data that will be used as the context in the prompt to the LLM. In order to do this search, context information will be converted to embeddings and stored in an AOSS collection. An embedding is a numerical representation of a given text in a vector space.

### Repository structure

This repository contains

* [A Jupyter Notebook](https://github.com/aws-samples/reinvent2023-aim-329/blob/main/notebooks/aim329.ipynb) to get started.

* [A set of helper functions for the notebook](https://github.com/aws-samples/reinvent2023-aim-329/blob/main/notebooks/scripts/helper_functions.py)

* [An architecture diagram](https://github.com/aws-samples/reinvent2023-aim-329/blob/main/notebooks/images/architecture.png) that shows the various components used in this session along with their interactions.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
