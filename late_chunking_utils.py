from transformers import AutoModel, AutoTokenizer
import numpy as np
import requests

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True)

def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations

def chunk_by_tokenizer_api(input_text: str):
    # Define the API endpoint and payload
    url = 'https://tokenize.jina.ai/'
    payload = {
        "content": input_text,
        "return_chunks": "true",
        "max_chunk_length": "1000"
    }

    # Make the API request
    response = requests.post(url, json=payload)
    response_data = response.json()

    # Extract chunks and positions from the response
    chunks = response_data.get("chunks", [])
    chunk_positions = response_data.get("chunk_positions", [])

    # Adjust chunk positions to match the input format
    span_annotations = [(start, end) for start, end in chunk_positions]

    return chunks, span_annotations

def late_chunking(
    model_output, span_annotation: list, max_length=None
):
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if (
            max_length is not None
        ):  # remove annotations which go bejond the max-length of the model
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs

def get_late_chunking_embeddings(text: str):
    inputs = tokenizer(text, return_tensors='pt')
    model_output = model(**inputs)
    
    # Using chunk_by_sentences for simplicity, can be replaced with chunk_by_tokenizer_api
    _, span_annotations = chunk_by_sentences(text, tokenizer)
    
    embeddings = late_chunking(model_output, [span_annotations])[0]
    return embeddings
