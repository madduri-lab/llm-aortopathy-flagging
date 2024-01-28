import time
import torch
from tqdm import tqdm
from itertools import islice, cycle
from utils.memory_utils import MemoryTrace
from utils.model_utils import save_peft_model

def train(
    model, 
    train_config,
    train_dataloader,
    eval_dataloader, 
    optimizer, 
    lr_scheduler, 
):
    """
    Trains the Causal LLM model on the given dataloader
    
    Args:
        model: The model to be trained
        train_config: The training configuration
        train_dataloader: The dataloader containing the training data
        eval_dataloader: The dataloader containing the eval data if doing evaluation
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        
    Returns: results dictionary containing average training and validation perplexity and loss
    """
    train_perp = []
    train_loss = []
    val_perp = []
    val_loss =[]
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            if train_config.max_train_batches == -1:
                total_length = len(train_dataloader)//train_config.gradient_accumulation_steps
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)
                for step, batch in enumerate(train_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(train_config.device)              
                    loss = model(**batch).loss
                    loss = loss / train_config.gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    loss.backward()
                    if (step + 1) % train_config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(step//train_config.gradient_accumulation_steps)
                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step+1}/{len(train_dataloader)} completed (loss: {loss.detach().float():.4f})")
            else:
                total_length = train_config.max_train_batches//train_config.gradient_accumulation_steps
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch}", total=total_length)
                max_train_batches = train_config.max_train_batches
                start_idx = (epoch * max_train_batches) % len(train_dataloader)
                train_iterator = cycle(train_dataloader)
                train_iterator = islice(train_iterator, start_idx, None)
                for step in range(max_train_batches):
                    batch = next(train_iterator)
                    for key in batch.keys():
                        batch[key] = batch[key].to(train_config.device)              
                    loss = model(**batch).loss
                    loss = loss / train_config.gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    loss.backward()
                    if (step + 1) % train_config.gradient_accumulation_steps == 0 or step == max_train_batches - 1:
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(step//train_config.gradient_accumulation_steps)
                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step+1}/{max_train_batches} completed (loss: {loss.detach().float():.4f})")
                
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        train_perp.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        print(f"\nMax CUDA memory allocated was {memtrace.peak} GB")
        print(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
        print(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
        print(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
        print(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epcoh time {epoch_end_time}s")
        
        # Update the learning rate as needed
        lr_scheduler.step()
          
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss = evaluate(model, eval_dataloader, train_config.device, train_config.max_val_batches)
            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and eval_epoch_loss < best_val_loss:
                save_peft_model(model, train_config.output_name)
                print(f"PEFT modules are saved at {train_config.output_name}")
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(best_val_loss)
            val_perp.append(eval_ppl)
        
    avg_train_prep = sum(train_perp)/len(train_perp)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    if train_config.run_validation:
        avg_eval_prep = sum(val_perp)/len(val_perp) 
        avg_eval_loss = sum(val_loss)/len(val_loss) 
        avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
        results["avg_checkpoint_time"] = avg_checkpoint_time
    return results

def evaluate(model, eval_dataloader, device, max_val_batches):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        device: The device for running evaluation
    
    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    eval_loss = 0.0  # Initialize evaluation loss
    with MemoryTrace():
        for step, batch in enumerate(eval_dataloader):
            if max_val_batches > 0:
                if step >= max_val_batches:
                    break
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                eval_loss += loss.detach().float()
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader) if max_val_batches <= 0 else eval_loss / max_val_batches
    eval_ppl = torch.exp(eval_epoch_loss)
        
    return eval_ppl, eval_epoch_loss