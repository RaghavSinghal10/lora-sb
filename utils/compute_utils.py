import numpy as np

def create_compute_metrics_function(tokenizer, args, metric):
    """Create a compute_metrics function with the required parameters."""
    
    task_to_labels = {
        'cola': ['unacceptable', 'acceptable'],
        'sst2': ['negative', 'positive'],
        'mrpc': ['not_equivalent', 'equivalent'],
        'qqp': ['not_duplicate', 'duplicate'],
        'stsb': None,  # Regression task
        'mnli': ['entailment', 'neutral', 'contradiction'],
        'qnli': ['entailment', 'not_entailment'],
        'rte': ['entailment', 'not_entailment'],
        'wnli': ['not_entailment', 'entailment']
    }
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        
        # Convert predictions to numpy array and get the first element if it's a tuple
        if isinstance(preds, tuple):
            preds = preds[0]
        
        # Ensure preds is a numpy array
        preds = np.array(preds)
        
        # Handle multi-dimensional arrays (e.g., from beam search)
        if len(preds.shape) > 2:
            preds = preds.reshape(-1, preds.shape[-1])
        
        # Convert to list of lists if necessary
        if isinstance(preds[0], np.ndarray):
            preds = [pred.tolist() for pred in preds]
        
        try:
            # Decode predictions
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            # Clean up decoded predictions (remove extra whitespace)
            decoded_preds = [pred.strip() for pred in decoded_preds]
            
            if args.task == 'stsb':
                # Convert string predictions to float for regression task
                # Handle potential formatting issues
                decoded_preds = [float(pred.replace(',', '.')) if pred.strip() else 0.0 
                               for pred in decoded_preds]
            else:
                # Convert string predictions to label indices for classification tasks
                label_list = task_to_labels[args.task]
                decoded_preds = [label_list.index(pred) if pred in label_list 
                               else 0 for pred in decoded_preds]

            # Handle labels
            if labels is not None:
                # Replace -100 with pad token
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                
                # Convert labels to list of lists if necessary
                if isinstance(labels[0], np.ndarray):
                    labels = [label.tolist() for label in labels]
                
                # Decode reference labels
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_labels = [label.strip() for label in decoded_labels]
                
                if args.task == 'stsb':
                    # Convert string labels to float for regression task
                    decoded_labels = [float(label.replace(',', '.')) if label.strip() else 0.0 
                                    for label in decoded_labels]
                else:
                    # Convert string labels to label indices for classification tasks
                    decoded_labels = [label_list.index(label) if label in label_list 
                                    else 0 for label in decoded_labels]

            # Compute metrics
            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            
            # Handle potential None results
            if result is None:
                return {"accuracy": 0.0}
                
            return result
            
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            print(f"Sample predictions shape: {preds.shape if hasattr(preds, 'shape') else 'no shape'}")
            print(f"Sample predictions: {preds[:5]}")
            return {"accuracy": 0.0}
    
    return compute_metrics