# Ablations for Gemma 7B:
python ablation-probability-hazard.py google/gemma-1.1-7b-it ablations/
python ablation-saliency-hazard.py google/gemma-1.1-7b-it ablations/

# Ablations for Gemma 2 9B:
python ablation-probability-hazard.py google/gemma-2-9b-it ablations/
python ablation-saliency-hazard.py google/gemma-2-9b-it ablations/

# Ablations for Llama 3.1 8B:
python ablation-probability-hazard.py meta-llama/Meta-Llama-3.1-8B-Instruct ablations/
python ablation-saliency-hazard.py meta-llama/Meta-Llama-3.1-8B-Instruct ablations/

# Experiments for Gemma 2B:
python self-explanations-hazard.py google/gemma-1.1-2b-it -p -s
python self-counterfactuals-hazard.py google/gemma-1.1-2b-it 5 1. -p
python self-explanations-movies.py google/gemma-1.1-2b-it -s
python self-counterfactuals-movies.py google/gemma-1.1-2b-it 5 1. -p

# Experiments for Gemma 7B:
python self-explanations-hazard.py google/gemma-1.1-7b-it -p -s
python self-counterfactuals-hazard.py google/gemma-1.1-7b-it 5 1. -p
python self-explanations-movies.py google/gemma-1.1-7b-it -s
python self-counterfactuals-movies.py google/gemma-1.1-7b-it 5 1. -p

# Experiments for Gemma 2 9B:
python self-explanations-hazard.py google/gemma-2-9b-it -p -s
python self-counterfactuals-hazard.py google/gemma-2-9b-it 5 1. -p
python self-explanations-movies.py google/gemma-2-9b-it -s
python self-counterfactuals-movies.py google/gemma-2-9b-it 5 1. -p

# Experiments for Gemma 2 27B:
##python self-explanations-hazard.py google/gemma-2-27b-it
python self-counterfactuals-hazard.py google/gemma-2-27b-it 5 1. -p
##python self-explanations-movies.py google/gemma-2-27b-it
python self-counterfactuals-movies.py google/gemma-2-27b-it 5 1. -p

# Experiments for Llama 3.1 8B:
python self-explanations-hazard.py meta-llama/Meta-Llama-3.1-8B-Instruct -p -s
python self-counterfactuals-hazard.py meta-llama/Meta-Llama-3.1-8B-Instruct 5 1. -p
python self-explanations-movies.py meta-llama/Meta-Llama-3.1-8B-Instruct -s
python self-counterfactuals-movies.py meta-llama/Meta-Llama-3.1-8B-Instruct 5 1. -p

# Experiments for Llama 3.1 70B:
##python self-explanations-hazard.py meta-llama/Meta-Llama-3.1-70B-Instruct -p -s
python self-counterfactuals-hazard.py meta-llama/Meta-Llama-3.1-70B-Instruct 5 1. -p
##python self-explanations-movies.py meta-llama/Meta-Llama-3.1-70B-Instruct -s
python self-counterfactuals-movies.py meta-llama/Meta-Llama-3.1-70B-Instruct 5 1. -p

# Experiments for Llama 3.2 1B:
python self-explanations-hazard.py meta-llama/Llama-3.2-1B-Instruct -p -s
python self-counterfactuals-hazard.py meta-llama/Llama-3.2-1B-Instruct 5 1. -p
python self-explanations-movies.py meta-llama/Llama-3.2-1B-Instruct -s
python self-counterfactuals-movies.py meta-llama/Llama-3.2-1B-Instruct 5 1. -p

# Experiments for Llama 3.2 3B:
python self-explanations-hazard.py meta-llama/Llama-3.2-3B-Instruct -p -s
python self-counterfactuals-hazard.py meta-llama/Llama-3.2-3B-Instruct 5 1. -p
python self-explanations-movies.py meta-llama/Llama-3.2-3B-Instruct -s
python self-counterfactuals-movies.py meta-llama/Llama-3.2-3B-Instruct 5 1. -p

# Experiments for Llama 3.3 70B:
##python self-explanations-hazard.py meta-llama/Llama-3.3-70B-Instruct -p -s
python self-counterfactuals-hazard.py meta-llama/Llama-3.3-70B-Instruct 5 1. -p
##python self-explanations-movies.py meta-llama/Llama-3.3-70B-Instruct -s
python self-counterfactuals-movies.py meta-llama/Llama-3.3-70B-Instruct 5 1. -p