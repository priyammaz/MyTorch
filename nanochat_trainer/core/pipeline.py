"""
Inference will be handled by a Pipeline kind of like Huggingface Pipelines, lots of inspiration
from nanoChat of course!
https://github.com/karpathy/nanochat/blob/master/nanochat/engine.py#L228
"""

import mytorch
import mytorch.nn.functional as F
from collections import deque
from .cache import KVCache
from .tools import calculator

class Pipeline:

    def __init__(self, 
                 model, 
                 tokenizer,
                 device="cuda"):

        assert "cuda" in device, "Model requires CUDA for inference as its uses Fused ops!"
        
        ### Load Model and throw into evaluation mode ###
        self.model = model.to(device)
        if not hasattr(model, "config"):
            raise Exception("No config found in model!!!")
        self.config = model.config
        self.model.eval()   

        ### Store Tokenizer ###
        self.tokenizer = tokenizer

        ### Store device ###
        self.device = device

    def get_cache(self, batch_size):
        return KVCache(
            batch_size=batch_size, 
            seq_len=self.config.sequence_length,
            num_heads=self.config.num_kv_heads, 
            head_dim=self.config.embed_dim // self.config.num_kv_heads,
            num_layers=self.config.num_blocks 
        )

    def per_row_generation_meta(self, tokens=None):
        
        """
        Different samples in our generation can end at different times, lets store some metadata
        about each generation, this is equivalent to the RowState from nanoChat!
        """
        assert isinstance(tokens, list), "Tokens must be a list!"
        
        return {
            "tokens": tokens if tokens is not None else [],
            "forced_tokens": deque(),
            "is_python": False, 
            "python_expr": [],
            "completed": False
        }
    
    @staticmethod
    @mytorch.no_grad()
    def sample(logits, temperature=1.0, topk=None):
        """
        sample logits where the shape is [B x Vocab Size]
        and predict the next token!
        """
        if temperature == 0.0:
            ### grab the per row argmax, if temp is 0 it is deterministic ###
            next_token_pred = mytorch.argmax(logits, dim=-1).unsqueeze(1)
        else:
            if topk is not None:
                
                ### Get logit indexes from high to low ###
                high_to_low_logits = logits.argsort(dim=-1, descending=True)
                topk_logits_idx = high_to_low_logits[:, :topk]
                batch_idx = mytorch.arange(logits.shape[0]).unsqueeze(1)

                ### Get the corresponding logits ###
                logits_subset = logits[batch_idx, topk_logits_idx]

                ### Scale logits with temperature ###
                logits_subset = logits_subset / temperature

                ### Take the softmax over the topk subset ###
                probs = F.softmax(logits_subset, dim=-1)
                
                ### Sample from the distribution ###
                pred_idx = mytorch.multinomial(probs)

                ### Grab the pred_idx from our actual token indexes ###
                next_token_pred = topk_logits_idx[batch_idx, pred_idx]
            
            else:
                
                ### Scale by Temperature ###
                logits = logits / temperature

                ### Compute Probs ###
                probs = F.softmax(logits, dim=-1)

                ### Sample ###
                next_token_pred = mytorch.multinomial(probs)
        
        return next_token_pred
    
    @mytorch.no_grad()
    def _generate(self,
                  input_ids, 
                  num_generations=1, 
                  max_token_gens=None, 
                  temperature=1.0, 
                  topk=None):
        
        if isinstance(input_ids, list):
            input_ids = mytorch.Tensor(input_ids).reshape(1,-1)
        input_ids = input_ids.to(self.device)

        assert len(input_ids.shape) == 2, "Input ids must be [Batch Size x Seq Len] !!"
        assert input_ids.shape[0] == 1, f"Only support single sample input generations. got {input_ids.shape[0]} samples"

        ### Initialize KV Cache ###
        cache = self.get_cache(1)

        ### Do our first forward pass to get our initial logits ###
        logits, cache = self.model(input_ids, cache=cache)
        
        ### Expand out for however many generations we want ###
        cache.repeat(num_generations)
        logits = mytorch.concatenate([logits for _ in range(num_generations)], dim=0)

        ### Grab the last timesteps logits so we can predict the next token ###
        logits = logits[:, -1, :]
        
        ### Sample input ids (this is now (b x 1)) ###
        new_input_ids = self.sample(logits, temperature, topk)

        ### Start row states ###
        input_ids_list = input_ids[0].numpy().tolist() # all of our input tokens 
        new_input_ids_list = new_input_ids.flatten().numpy().tolist() # our next token (per generation)
        states = [self.per_row_generation_meta(input_ids_list + [new_input_ids_list[i]]) for i in range(num_generations)] 
        
        ### Start Generations ###
        input_ids = new_input_ids
        num_generated_tokens = 0

        while True:

            ### Stop if we hit our max length ###
            if max_token_gens is not None and num_generated_tokens >= max_token_gens:
                break
            
            ### Stop if all our row states are flagged as done ###
            if all(s["completed"] for s in states):
                break
            
            ### Sample next token ###
            logits, cache = self.model(input_ids, cache=cache)
            logits = logits[:, -1, :]
            input_ids = self.sample(logits, temperature, topk)

            ### List of our per sample next token prediction ###
            input_ids_list = input_ids.flatten().numpy().tolist()
            
            ### Now we start to process our predicted tokens ###
            actual_next_tokens = [] # store the actual next token per sample
            forced_mask = []
            for i in range(len(input_ids_list)):
                
                ### Forced Tokens are tokens we have to add after a specific trigger ###
                ### for example when we complete a python block and our calculator ###
                ### does its operation and returns its answer, we need to force our ###
                ### next tokens to be "<|output_start|>" + [ANSWER] + "<|output_end|>" ###
                ### But we are generating potentially multiple samples at once, and if ###
                ### things have different lengths it would be annoying. So if we have cached ###
                ### some forced tokens, then regardless of whatever our next token pred is ###
                ### we will just add in these forced tokens until we are out and then keep going ###
                ### like normal!! ##

                state = states[i]
                next_pred_token = input_ids_list[i]

                ### Check if we have any forced tokens ###
                is_forced = (len(state["forced_tokens"]) > 0)
                forced_mask.append(0 if is_forced else 1)

                ### Grab either the forced token if we have any or our actual next pred token from the model ###
                next_token = state["forced_tokens"].popleft() if is_forced else next_pred_token
                actual_next_tokens.append(next_token)

                ### Add the actual next tokens to our state as well ###
                state["tokens"].append(next_token)

                ### For our pretrained model we have a <|bos|> token as our delimiter between samples ###
                ### and when finetuned the end of the assistant is <|assistant_end|> so either of these ###
                ### will trigger our end of generation for this specific sample ###
                if (next_token == self.tokenizer.assistant_end_id) or (next_token == self.tokenizer.bos_token_id):
                    state["completed"] = True 
                
                ### This is if the LLM has triggered a python start block ###
                if next_token == self.tokenizer.python_start_id:
                    state["is_python"] = True
                    state["python_expr"] = []
                
                ### If the next token is that we are ending the python block and we were in one ###
                elif (next_token == self.tokenizer.python_end_id) and state["is_python"]:       
                    
                    ### We are done then with python ###
                    state["is_python"] = False       

                    ### If we have some expression (in token form for now) ###
                    if len(state["python_expr"]) > 0:
                        
                        ### Get the expression (tokens to the actual text) ###
                        expression = self.tokenizer.decode(state["python_expr"])

                        ### Use our calculator on this expression ###
                        result = calculator(expression)

                        ### If we got a successful result ###
                        if result is not None:
                            
                            ### Retokenize the answer ###
                            result_tokens = self.tokenizer.encode(str(result))

                            ### Cache some forced tokens for the "<|output_start|>" + [ANSWER] + "<|output_end|>" ###
                            state["forced_tokens"].append(self.tokenizer.output_start_id)
                            state["forced_tokens"].append(result_tokens)
                            state["forced_tokens"].append(self.tokenizer.output_end_id)

                    ### Reset our expression ###
                    state["python_expr"] = []
                
                ### If we are in a python block just keep adding our tokens in ###
                elif state["is_python"]:
                    state["python_expr"].append(next_token)

            yield actual_next_tokens, forced_mask

            num_generated_tokens += 1
            input_ids = mytorch.Tensor(actual_next_tokens, dtype=mytorch.int32).reshape(-1,1)
            
    def generate(self,
                 input_ids, 
                 num_generations=1, 
                 max_token_gens=None, 
                 temperature=1.0, 
                 topk=None):

        if isinstance(input_ids, list):
            input_ids = mytorch.Tensor(input_ids).reshape(1,-1)
        input_ids = input_ids.to(self.device)

        assert len(input_ids.shape) == 2, "Input ids must be [Batch Size x Seq Len] !!"
        assert input_ids.shape[0] == 1, f"Only support single sample input generations. got {input_ids.shape[0]} samples"

        generator = self._generate(
            input_ids, 
            num_generations, 
            max_token_gens, 
            temperature, 
            topk
        )

        ### Create a list of lists as our starting point ###
        output = [input_ids[0].numpy().tolist().copy() for _ in range(num_generations)]
        
        ### Input tokens are masked ###
        masks = [[0]*len(output[0]) for _ in range(num_generations)]

        ### Completion flag ###
        completed = [False for _ in range(num_generations)]

        for next_tokens, next_mask in generator:
            
            ### This gives us the next token and mask val per generation ###
            for i, (token_, mask_) in enumerate(zip(next_tokens, next_mask)):

                ### If we are not done ###
                if not completed[i]:

                    ### If the next token is a done token ###
                    if (token_ == self.tokenizer.assistant_end_id) or (token_ == self.tokenizer.bos_token_id):
                        completed[i] = True

                    ### otherwise just keep going ###
                    else:
                        output[i].append(token_)
                        masks[i].append(mask_)
                
            # Stop if all rows are completed 
            if all(completed):
                break
        
        return output, masks
                        
if __name__ == "__main__":

    from .nanochat_gpt import GPT, GPTConfig
    from .tokenizer import MyTokenizer

    model = GPT(GPTConfig(num_blocks=2))
    tokenizer = MyTokenizer()
    pipe = Pipeline(model, tokenizer)

    rand = mytorch.randint(0,100, shape=(1,4))
    pipe.generate(rand, num_generations=2, max_token_gens=100)