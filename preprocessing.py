import json
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def preprocess_V1(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

     # Collect all unique combinations of capability and test_type
    combinations = {(item['capability'], item['test_type']) for item in data}
    
    for capability, test_type in combinations:
        conllu_output = []
        for item in filter(lambda x: x['capability'] == capability and x['test_type'] == test_type, data):
            sentence_id = item['sentence_id']
            sentence = item['sentence']
            predicate_id = item['predicate_id']
        
            # Tokenize the sentence using NLTK
            tokens = word_tokenize(sentence)
            predicate = tokens[predicate_id - 1]
        
            # Prepare CoNLL-U format output
            for i, token in enumerate(tokens, start=1):
                label = '_'  # Default label if token doesn't match any tokenX
                for key in item.keys():
                    if key.startswith('token') and token == item[key]:
                        label_key = key.replace('token', 'expected')
                        if label_key in item:
                            label = item[label_key]
                            break
                
                conllu_output.append(f"{sentence_id}\t{i}\t{token}\t{label}")
            
            # Add predicate after [SEP] token
            conllu_output.append(f"{sentence_id}\t{i+1}\t[SEP]\t_")
            conllu_output.append(f"{sentence_id}\t{i+2}\t{predicate}\t_")
            conllu_output.append('')  # Add an empty line between sentences
        
        # Write output to a file
        output = f'datasets/input_M1/{capability}_{test_type}_M1.conllu'
        with open(output, 'w') as f:
            f.write('\n'.join(conllu_output))

def preprocess_V2(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Collect all unique combinations of capability and test_type
    combinations = {(item['capability'], item['test_type']) for item in data}

    for capability, test_type in combinations:
        conllu_output = []
        for item in filter(lambda x: x['capability'] == capability and x['test_type'] == test_type, data):
            sentence_id = item['sentence_id']
            sentence = item['sentence']
            predicate_id = item['predicate_id']
        
            # Tokenize the sentence using NLTK
            tokens = word_tokenize(sentence)
            predicate = tokens[predicate_id - 1]
        
            # Find the position of the predicate in the tokens
            predicate_index = tokens.index(predicate) + 1
            
            # Prepare CoNLL-U format output
            for i, token in enumerate(tokens, start=1):
                label = '_'  # Default label if token doesn't match any tokenX
                for key in item.keys():
                    if key.startswith('token') and token == item[key]:
                        label_key = key.replace('token', 'expected')
                        if label_key in item:
                            label = item[label_key]
                            break  # Stop searching if token found
                
                conllu_output.append(f"{sentence_id}\t{i}\t{token}\t{label}")
            
            # Add [SEP] token
            conllu_output.append(f"{sentence_id}\t{i+1}\t[SEP]\t_")
        
            # Add preceding token, predicate, and token after predicate
            for j in range(predicate_index - 2, predicate_index + 1):
                if j >= 0 and j < len(tokens):
                    token = tokens[j]
                    label = '_'
                    for key in item.keys():
                        if key.startswith('token') and token == item[key]:
                            label_key = key.replace('token', 'expected')
                            if label_key in item:
                                label = item[label_key]
                                break
                    conllu_output.append(f"{sentence_id}\t{i+2}\t{token}\t_")
                    i += 1
            
            conllu_output.append('')  # Add an empty line between sentences
    
        # Write output to a file
        output = f'datasets/input_M2/{capability}_{test_type}_M2.conllu'
        with open(output, 'w') as f:
            f.write('\n'.join(conllu_output))

def preprocess_V3(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

     # Collect all unique combinations of capability and test_type
    combinations = {(item['capability'], item['test_type']) for item in data}
    
    for capability, test_type in combinations:
        conllu_output = []
        for item in filter(lambda x: x['capability'] == capability and x['test_type'] == test_type, data):
            sentence_id = item['sentence_id']
            sentence = item['sentence']
            predicate_id = item['predicate_id']
        
            # Tokenize the sentence using NLTK
            tokens = word_tokenize(sentence)
            predicate = tokens[predicate_id - 1]
            predicate_encountered = False
            
            # Prepare CoNLL-U format output
            for i, token in enumerate(tokens, start=1):
                # Marking the predicate with the special token if it's the first matching token
                if token == predicate and not predicate_encountered:
                    token = '[PRED] ' + token 
                    predicate_encountered = True 
                label = '_'  # Default label if token doesn't match any tokenX
                for key in item.keys():
                    if key.startswith('token') and token == item[key]:
                        label_key = key.replace('token', 'expected')
                        if label_key in item:
                            label = item[label_key]
                            break
                
                conllu_output.append(f"{sentence_id}\t{i}\t{token}\t{label}")
            conllu_output.append('')  # Add an empty line between sentences

        # Write output to a file
        output = f'datasets/input_M3/{capability}_{test_type}_M3.conllu'
        with open(output, 'w') as f:
            f.write('\n'.join(conllu_output))

def process_all_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            json_file_path = os.path.join(directory, filename)
            preprocess_V1(json_file_path)
            preprocess_V2(json_file_path)
            preprocess_V3(json_file_path)

if __name__ == "__main__":
    datasets_directory = "datasets"
    process_all_json_files(datasets_directory)



