import os

def evaluate_mft(file_path):
    """
    Evaluates Minimum Functionality tests (MFT) on a labeled dataset to check if system labels match gold labels for each token.
    Parameters:
    - file_path (str): The path to the file containing the dataset with system predictions.
    Returns:
    - tuple: Contains the failure rate and a list of sentence IDs that failed the MFT test.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        current_sentence_id = None
        sentence_labels_match = True
        total_sentences = 0
        failed_sentences = 0
        failed_sentence_ids = []

        for line in file:
            # Skip empty lines
            if line.strip() == "":
                if current_sentence_id is not None:
                    total_sentences += 1
                    if not sentence_labels_match:
                        failed_sentences += 1
                        failed_sentence_ids.append(current_sentence_id)
                    # Reset for the next sentence
                    sentence_labels_match = True
                continue

            sentence_id, token_id, token, gold_label, system_label = line.strip().split('\t')

            # If we're still on the same sentence, check the labels
            if sentence_id == current_sentence_id:
                if gold_label != "_" and gold_label != system_label:
                    sentence_labels_match = False
            else:
                # If this is a new sentence, first check if the previous sentence failed
                if current_sentence_id is not None:
                    if not sentence_labels_match:
                        failed_sentences += 1
                        failed_sentence_ids.append(current_sentence_id)

                current_sentence_id = sentence_id
                sentence_labels_match = True if gold_label == "_" or gold_label == system_label else False

    failure_rate = (failed_sentences / total_sentences) * 100
    return int(failure_rate), failed_sentence_ids

def evaluate_inv(file_path):
    """
    Evaluates Invariance tests (INV) by comparing pairs of sentences to ensure that system labels for tokens with relevant gold labels (not '_') do not change between the sentence pairs regardless of perturbations.
    Parameters:
    - file_path (str): The path to the file containing the dataset with system predictions.
    Returns:
    - tuple: Contains the failure rate (as a percentage) and a list of sentence pair IDs that failed the INV test.
    """
    sentences_data = {}
    total_pairs = 0
    failed_pairs = 0
    failed_sentence_ids = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                continue

            sentence_id, token_id, token, gold_label, system_label = line.strip().split('\t')
            sentence_id = int(sentence_id)
            
            # Skip tokens with gold label '_'
            if gold_label == "_":
                continue
            
            # If it's a new sentence ID, initialise it with an empty list
            if sentence_id not in sentences_data:
                sentences_data[sentence_id] = []
            
            # Store both gold and system labels for tokens with gold labels other than '_'
            sentences_data[sentence_id].append((gold_label, system_label))

        sentence_ids = sorted(sentences_data.keys())
        # Making sure that the number of input sentences is even
        if len(sentence_ids) % 2 != 0: 
            print('Warning: uneven number of sentences!')

        # Evaluating sentences in pairs
        for i in range(0, len(sentence_ids), 2):
            first_sentence_data = sentences_data[sentence_ids[i]]
            second_sentence_data = sentences_data[sentence_ids[i+1]]
            total_pairs += 1
            pair_failed = False

            # Map gold labels to system labels for easier comparison
            first_sentence_map = {gold: system for gold, system in first_sentence_data}
            second_sentence_map = {gold: system for gold, system in second_sentence_data}

            # Iterate over tokens in the first sentence
            for gold_label, system_label in first_sentence_data:
                # Check if the same gold label exists in the second sentence and focus on comparing system labels
                if gold_label in second_sentence_map and system_label != second_sentence_map[gold_label]:
                    pair_failed = True # Flag it as true if there is a change in system labels
                    break
            
            if pair_failed: # Update counter and store failed sentence IDs.
                failed_pairs += 1
                # failed_sentence_ids.extend([sentence_ids[i], sentence_ids[i+1]])
                failed_sentence_ids.append(f"{sentence_ids[i]}-{sentence_ids[i+1]}")

    failure_rate = (failed_pairs / total_pairs) * 100
    return int(failure_rate), failed_sentence_ids

def evaluate_dir(file_path):
    """
    Evaluates Directional Expectation tests (DIR) by comparing pairs of sentences to ensure that system labels for tokens with relevant gold labels (not '_') do change between the sentence pairs due to perturbations.
    Parameters:
    - file_path (str): The path to the file containing the dataset with system predictions.
    Returns:
    - tuple: Contains the failure rate (as a percentage) and a list of sentence pair IDs that failed the DIR test.
    """
    sentences_data = {}
    total_pairs = 0
    failed_pairs = 0
    failed_sentence_ids = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip() == "":
                continue

            sentence_id, token_id, token, gold_label, system_label = line.strip().split('\t')
            sentence_id = int(sentence_id)
            
            # Skip tokens with gold label '_'
            if gold_label == "_":
                continue
            
            if sentence_id not in sentences_data:
                sentences_data[sentence_id] = []
            
            sentences_data[sentence_id].append((token, gold_label, system_label))

        sentence_ids = sorted(sentences_data.keys())
        if len(sentence_ids) % 2 != 0: 
            print('Warning: uneven number of sentences!')

        for i in range(0, len(sentence_ids), 2):
            first_sentence_data = sentences_data[sentence_ids[i]]
            second_sentence_data = sentences_data[sentence_ids[i+1]]
            total_pairs += 1
            pair_failed = False

            first_sentence_map = {token: (gold, system) for token, gold, system in first_sentence_data}
            second_sentence_map = {token: (gold, system) for token, gold, system in second_sentence_data}

            # Iterate over tokens in the first sentence
            for token, (gold_label, system_label) in first_sentence_map.items():
                # Check if the same token exists in the second sentence
                if token in second_sentence_map:
                    _, second_system_label = second_sentence_map[token]
                    # If the system label does not change, mark the pair as failed
                    if system_label == second_system_label:
                        pair_failed = True
                        break
            
            if pair_failed:
                failed_pairs += 1
                failed_sentence_ids.append(f"{sentence_ids[i]}-{sentence_ids[i+1]}")

    failure_rate = (failed_pairs / total_pairs) * 100
    return int(failure_rate), failed_sentence_ids

def evaluate_file(file_path):
    # Determine the type of evaluation based on the filename
    if 'MFT' in file_path:
        failure_rate, failed_sentence_ids = evaluate_mft(file_path)
        evaluation_type = 'MFT'
    elif 'INV' in file_path:
        failure_rate, failed_sentence_ids = evaluate_inv(file_path)
        evaluation_type = 'INV'
    elif 'DIR' in file_path:
        failure_rate, failed_sentence_ids = evaluate_dir(file_path)
        evaluation_type = 'DIR'
    else:
        print(f"Warning: could not determine evaluation type for file: {file_path}")

    # Print the evaluation results
    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"Evaluation Type: {evaluation_type}")
    print(f"Failure Rate: {failure_rate}%")
    if 'MFT' in file_path and failed_sentence_ids:
        print(f"Failed Sentence IDs: {failed_sentence_ids}")
    elif failed_sentence_ids:
        print(f"Failed Sentence Pairs: {', '.join(failed_sentence_ids)}")


def main():
    # List of directories to evaluate
    output_dirs = ['datasets/output_M1', 'datasets/output_M2', 'datasets/output_M3']

    for output_dir in output_dirs:
        print(f"Evaluating model: {output_dir}")
        # Get all file names, sort them alphabetically, and filter out directories
        file_names = sorted([f for f in os.listdir(output_dir) if not os.path.isdir(os.path.join(output_dir, f))])

        for filename in file_names:
            # Construct the full file path
            file_path = os.path.join(output_dir, filename)
            if filename.endswith('.tsv'):
                evaluate_file(file_path)
        print('-------------------------------------------------------\n')

if __name__ == "__main__":
    main()