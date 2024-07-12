#!/bin/bash

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
    echo "No training argument provided"
    exit 1
fi

# Loop over each argument
for file in "$@"; do
	# Check if the file exists
	if [ -e "$file" ]; then
		echo "Now running training config: $file"
		python3 finetune_ALBERT_by_sequence_classification.py $file
		python3 finetune_BART_by_sequence_classification.py $file
		python3 finetune_BERT_by_sequence_classification.py $file
		python3 finetune_GPT2_by_sequence_classification.py $file
		python3 finetune_RoBERTa_by_sequence_classification.py $file
		python3 finetune_T5_by_sequence_classification.py $file
	else
		echo "File not found: $file"
	fi
done
