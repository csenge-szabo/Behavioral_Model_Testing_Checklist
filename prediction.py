from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import os

def determine_label(subtoken_labels):
    """
    Determines the most frequent label among subtokens of a token.
    Args: subtoken_labels (list of str): A list of labels for the subtokens of a single token.
    Returns: str: The most frequent label among the subtokens.
    """
    return max(set(subtoken_labels), key=subtoken_labels.count)

def bert_predictions(model, tokenizer, index_to_label, sentence_list, gold_list, output_path):
    """
    Process and evaluate sentences with a DistilBERT model, writing the results to an output tsv file.
    Args:
        model (PreTrainedModel): The pre-trained model for token classification.
        tokenizer (PreTrainedTokenizer): The tokenizer for the model.
        index_to_label (dict): Mapping from label indices to label names.
        sentence_list (list of lists of str): List of sentences, where each sentence is represented as a list of tokens.
        gold_list (list of lists of str): List of gold labels corresponding to each token in the sentences.
        output_path (str): Path to the output file where predictions and gold labels are stored.
    Returns: None
    """
    with open(output_path, 'w', encoding='utf-8') as output_file:
        sentence_id = 1
        # Process each sentence in the input
        for sentence, gold_labels in zip(sentence_list, gold_list):
            # Tokenize the sentence and prepare inputs for the model predictions
            inputs = tokenizer(sentence, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True,
                               return_offsets_mapping=True)
            # Remove offset mappings from the inputs
            offset_mapping = inputs.pop('offset_mapping')

            model.eval() # Set the model to evaluation mode

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            word_labels = [None] * len(sentence)
            current_word_index = 0
            subtoken_labels = []

            # Process each token and its corresponding prediction
            for idx, ((start, end), pred) in enumerate(zip(offset_mapping[0], predictions[0])):
                # Skip special tokens and padding tokens
                if start == end == 0:
                    continue

                label = index_to_label[pred.item()]

                # Determine the label for the current word based on its subtokens
                if start == 0 and subtoken_labels:
                    word_labels[current_word_index] = determine_label(subtoken_labels)
                    current_word_index += 1
                    subtoken_labels = []

                subtoken_labels.append(label)

            # Finalize the label for the last token in the sentence
            if subtoken_labels:
                word_labels[current_word_index] = determine_label(subtoken_labels)

            # Write the token, gold label, and predicted label to the output file
            for token_id, (token, system_label, gold_label) in enumerate(zip(sentence, word_labels, gold_labels)):
                if token == "[SEP]": # Stop at [SEP] token
                    break
                # Handle predicate tokens marked with [PRED]
                if '[PRED]' in token:
                    token = token.replace('[PRED] ', '')
                output_file.write(f"{sentence_id}\t{token_id+1}\t{token}\t{gold_label}\t{system_label}\n")
            output_file.write("\n")  # Separate sentences with an empty line
            sentence_id += 1

def read_sentences_from_file(file_path):
    """
    Reads sentences and their gold labels from a file.
    Args: file_path (str): Path to the file containing sentences and gold labels.
    Returns: tuple:
            - The first list contains sentences, where each sentence is represented as a list of tokens.
            - The second list contains the corresponding gold labels for each token in the sentences.
    """
    sentences = []
    gold_list = []
    current_sentence_tokens = []
    current_sentence_gold = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                # End of a sentence
                if current_sentence_tokens:
                    sentences.append(current_sentence_tokens)
                    gold_list.append(current_sentence_gold)
                    current_sentence_tokens = []
                    current_sentence_gold = []
            else:
                # Extract token and gold label from the line
                token, gold = line.strip().split('\t')[2:4]
                current_sentence_tokens.append(token)
                current_sentence_gold.append(gold)

        # Add the last sentence if file does not end with a newline
        if current_sentence_tokens:
            sentences.append(current_sentence_tokens)
            gold_list.append(current_sentence_gold)

    return sentences, gold_list

def run_model_predictions(model_name, input_dir, output_dir, label_to_index, index_to_label):
    """
    Runs prediction on all files within an input directory using a model, and outputs the results to an output directory.
    Args:
        - model_name (str): The name of the model to use for predictions.
        - input_dir (str): The directory containing input files to process.
        - output_dir (str): The directory where output files will be saved.
        - label_to_index (dict): Mapping from label names to label indices.
        - index_to_label (dict): Mapping from label indices to label names.
    Returns: None
    """
    model_path = f"models/{model_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.conllu'):
            file_path = os.path.join(input_dir, filename)
            print(f'Predicting file: {file_path}')

            output_file_name = filename.replace(".conllu", "_predictions.tsv")
            output_path = os.path.join(output_dir, output_file_name)

            # Read sentences and their gold labels from the input file
            active_sentence_list, gold_labels_list = read_sentences_from_file(file_path)

            # Perform predictions and write the results to the output file.
            bert_predictions(model, tokenizer, index_to_label, active_sentence_list, gold_labels_list, output_path)

def main():
    label_to_index = {'ARG0': 0, 'ARG1': 1, 'ARG1-DSP': 2, 'ARG2': 3, 'ARG3': 4, 'ARG4': 5, 'ARG5': 6, 'ARGA': 7,
                            'ARGM-ADJ': 8, 'ARGM-ADV': 9, 'ARGM-CAU': 10, 'ARGM-COM': 11, 'ARGM-CXN': 12, 'ARGM-DIR': 13,
                            'ARGM-DIS': 14, 'ARGM-EXT': 15, 'ARGM-GOL': 16, 'ARGM-LOC': 17, 'ARGM-LVB': 18, 'ARGM-MNR': 19,
                            'ARGM-MOD': 20, 'ARGM-NEG': 21, 'ARGM-PRD': 22, 'ARGM-PRP': 23, 'ARGM-PRR': 24, 'ARGM-REC': 25,
                            'ARGM-TMP': 26, 'C-ARG0': 27, 'C-ARG1': 28, 'C-ARG1-DSP': 29, 'C-ARG2': 30, 'C-ARG3': 31,
                            'C-ARG4': 32, 'C-ARGM-ADV': 33, 'C-ARGM-COM': 34, 'C-ARGM-CXN': 35, 'C-ARGM-DIR': 36,
                            'C-ARGM-EXT': 37, 'C-ARGM-GOL': 38, 'C-ARGM-LOC': 39, 'C-ARGM-MNR': 40, 'C-ARGM-PRP': 41,
                            'C-ARGM-PRR': 42, 'C-ARGM-TMP': 43, 'R-ARG0': 44, 'R-ARG1': 45, 'R-ARG2': 46, 'R-ARG3': 47,
                            'R-ARG4': 48, 'R-ARGM-ADJ': 49, 'R-ARGM-ADV': 50, 'R-ARGM-CAU': 51, 'R-ARGM-COM': 52,
                            'R-ARGM-DIR': 53, 'R-ARGM-GOL': 54, 'R-ARGM-LOC': 55, 'R-ARGM-MNR': 56, 'R-ARGM-TMP': 57, '_': 58}
    index_to_label = {v: k for k, v in label_to_index.items()}

    model_configs = {
        "BERT1_new": ("datasets/input_M1", "datasets/output_M1"),
        "BERT2_new": ("datasets/input_M2", "datasets/output_M2"),
        "BERT3_new": ("datasets/input_M3", "datasets/output_M3"),
    }

    for model_name, (input_dir, output_dir) in model_configs.items():
        print(f"\nProcessing with {model_name}")
        run_model_predictions(model_name, input_dir, output_dir, label_to_index, index_to_label)

if __name__ == "__main__":
    main()
