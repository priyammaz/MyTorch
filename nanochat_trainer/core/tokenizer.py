"""
After running scripts/train_tokenizer.py we will have a tokenizer.json, this just wraps
that an makes it a useable tokenizer
"""
from tokenizers import Tokenizer

class MyTokenizer:
    def __init__(self, path_to_tokenizer_json="nanochat_trainer/nanochat_tokenizer/tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(path_to_tokenizer_json)
            
        ### Make special tokens easily accessible ###
        self.bos_token = "<|bos|>" # bos will just be used as a seperator basically (both for start and end)
        self.bos_token_id = self.encode_special(self.bos_token)
        self.system_start = "<|system_start|>"
        self.system_start_id = self.encode_special(self.system_start)
        self.system_end = "<|system_end|>"
        self.system_end_id = self.encode_special(self.system_end)
        self.user_start = "<|user_start|>"
        self.user_start_id = self.encode_special(self.user_start)
        self.user_end = "<|user_end|>"
        self.user_end_id = self.encode_special(self.user_end)
        self.assistant_start = "<|assistant_start|>"
        self.assistant_start_id = self.encode_special(self.assistant_start)
        self.assistant_end = "<|assistant_end|>"
        self.assistant_end_id = self.encode_special(self.assistant_end)
        self.python_start = "<|python_start|>"
        self.python_start_id = self.encode_special(self.python_start)
        self.python_end = "<|python_end|>"
        self.python_end_id = self.encode_special(self.python_end)
        self.output_start = "<|output_start|>"
        self.output_start_id = self.encode_special(self.output_start)
        self.output_end = "<|output_end|>"
        self.output_end_id = self.encode_special(self.output_end)

    def encode(self, text, prepend=None, append=None, add_special_tokens=True, max_tokens=None):
        """Encode a single string into token IDs."""
        if prepend is not None:
            assert isinstance(prepend, str), "prepend must be a string"
            text = prepend + text
        if append is not None:
            assert isinstance(append, str), "append must be a string"
            text = text + append
        
        tokenized = self.tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

        if (max_tokens is not None) and (len(tokenized) > max_tokens):
            tokenized = tokenized[:max_tokens]
        
        return tokenized 

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token IDs back into a string."""
        if not isinstance(token_ids, list):
            raise Exception("Expected list of integers to decode!!!")
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(self, text, prepend=None, append=None, add_special_tokens=True, max_tokens=None):
        
        if not isinstance(text, list):
            text = [text]

        return [self.encode(t, prepend, append, add_special_tokens, max_tokens) for t in text]

    def batch_decode(self, batch_token_ids, skip_special_tokens=True):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch_token_ids]
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def encode_special(self, text):
        return self.tokenizer.encode(text).ids[0]
    
    def get_special_tokens_and_ids(self):
        return {t: self.encode_special(t) for t in self.get_special_tokens()}
    
    def id_to_token(self, id):
        return self.tokenizer.decode(id)
    
    def parse_conversation(self, conversation, max_tokens=None):

        """
        Instead of Jinja we will just manually parse conversations here. Conversations will take the standard
        form in most llms. Data will be provided as:

        For normal conversations:
        {
            "messages": [
                {"role": "system", "content": ...}
                {"role": "user", "content": ...},
                {"role": "assistant", "content": ...},
                ...
            ]
        }

        For tool calling (on GSM8K) the content can have multiple parts
        {
            "messages": [
                {"role": "system", "content": ...}
                {"role": "user", "content": ...},
                {"role": "user", "content": [
                    {"type": "python", "text": "...some expression to eval with python"},
                    {"type": "python_output", "text": "...result from the toolcall"}
                ]} 
            ]
        }

        """

        ### First we create a two lists, one that has our tokens and the other a binary ###
        ### mask that identifies which tokens are trainable and which are not ###
        ### We dont really need to train on (next token prediction) on some of the special tokens 
        ### (things like user/assistant start) as those are set by the template. 
        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            assert mask_val in [0,1]
            if isinstance(token_ids, int):
                token_ids = [token_ids]

            ### Extend our token ids ###
            ids.extend(token_ids)
            
            ### Store mask for each of those ids ###
            mask.extend([mask_val] * len(token_ids))
        
        
        ### Grab the list of dictionaries ###
        messages = conversation["messages"]

        ## Start our tokens with a BOS token ###
        add_tokens(self.bos_token_id, 0)

        has_system_message = False
        for i, message in enumerate(messages):
            
            ### Sanity checks ###
            ### first can be either user or system message. 
            ### after we toggle between user/assistant
            role = message["role"]
 
            if role == "system":
                if i != 0:
                    raise Exception("Found a system message not at the start of the conversation!!")
                has_system_message = True

            ### If the first message is system, then every odd message (1,3,5,...) will be user and 
            ### every even message (2,4,6,...) will be assistant. But if there is no system message 
            ### then its opposite, every even message (0,2,4...) will be user and every odd (1,3,5,...)
            ### will be assistant!
            else:
                if has_system_message:
                    expected_role = "user" if i % 2 == 1 else "assistant"
                else:
                    expected_role = "user" if i % 2 == 0 else "assistant"
                
                if role != expected_role:
                    raise Exception(f"Message {i} role mismatch: expected '{expected_role}', got '{role}'")
            
            ### content can be either a string (or potentially a list for the assistant) ###
            content = message["content"]

            ### If a system message ###
            if role == "system":
               
                tokens = self.encode(content)
                ### Dont need loss on system tokens, just on completions ###
                assert isinstance(content, str), "System message content can only be a string"
                
                add_tokens(self.system_start_id, 0)
                add_tokens(tokens, 0)
                add_tokens(self.system_end_id, 0)

            ### If a user message ###
            elif role == "user":

                ### Dont need loss on user tokens, just on completions ###
                assert isinstance(content, str), "System message content can only be a string"
                tokens = self.encode(content)
                add_tokens(self.user_start_id, 0)
                add_tokens(tokens, 0)
                add_tokens(self.user_end_id, 0)

            ### If an assistant message ###
            elif role == "assistant":
                
                ### Start off with start token ###
                add_tokens(self.assistant_start_id, 0)

                ### If content is a string just add it on ###
                ### These completions are trainable! ###
                if isinstance(content, str):
                    tokens = self.encode(content)
                    add_tokens(tokens, 1)

                ### Content can also be a list (for tool calling)
                ### each list will have another dictionary with "text" as its content
                ### as well as a type telling us what it is
                ### the types can be "text" for normal text, "python" which is what we want
                ### to evaluate with a tool, and lastly, "python_output" which is our expected 
                ### output of the tool

                ### "text" and "python" is trainable, but "python_output" will be handled by
                ### the tool so no need to compute loss on that
                elif isinstance(content, list):
                    
                    for c in content:
                        tokens = self.encode(c["text"])
                        if c["type"] == "text":
                            add_tokens(tokens, 1)
                        elif c["type"] == "python":
                            add_tokens(self.python_start_id, 1)
                            add_tokens(tokens, 1)
                            add_tokens(self.python_end_id, 1)
                        elif c["type"] == "python_output":
                            add_tokens(self.output_start_id, 0)
                            add_tokens(tokens, 0)
                            add_tokens(self.output_end_id, 0)
                        else:
                            raise ValueError(f"Unkown type {c["type"]}")
                
                else:
                    raise ValueError(f"Unknown content type {type(content)}")
                
                ### End asssistant with end token ###
                add_tokens(self.assistant_end_id, 1)

        ### Truncate text incase it goes over our max context limit ###
        if max_tokens is not None:
            ids = ids[:max_tokens]
            mask = mask[:max_tokens]

        return ids, mask

    def __repr__(self):
        return str(self.tokenizer)

if __name__ == "__main__":

    tokenizer = MyTokenizer()
    
    print("TEST NORMAL TEXT\n")
    text = "Hello! I want to be a helpful assistant"
    encode = tokenizer.encode(text, prepend=tokenizer.bos_token)
    decode = tokenizer.decode(encode, skip_special_tokens=False)
    print("Encoded:", encode)
    print("Decoded:", decode)

    print("TEST BATCH ENCODE/DECODE\n")
    texts = [text, text, text]
    encode = tokenizer.batch_encode(texts, prepend=tokenizer.bos_token)
    decode = tokenizer.batch_decode(encode, skip_special_tokens=False)
    print("Encoded:", encode)
    print("Decoded:", decode)

    print("\nTEST CONVERSATION")
    conversation = {
                "messages": [
                    {"role": "system", "content": "you are a helpful AI"},
                    {"role": "user", "content": "What is the color of the sky?"},
                    {"role": "assistant", "content": "the sky is blue!"},
                ]
            }



    ids, mask = tokenizer.parse_conversation(conversation)
    print("Encoded:", ids)
    print("Mask:", mask)
    decode = tokenizer.decode(ids, skip_special_tokens=False)
    print("Decoded:", decode)


    print("\nTEST TOOLCALL CONVERSATION")
    conversation = {
                "messages": [
                    {"role": "system", "content": "you are a helpful AI"},
                    {"role": "user", "content": "What is 7 times 2?"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "first we compute 7 * 2"},
                        {"type": "python", "text": "7*2"},
                        {"type": "python_output", "text": "14"}
                    ]},
                ]
            }



    ids, mask = tokenizer.parse_conversation(conversation)
    print("Encoded:", ids)
    print("Mask:", mask)
    decode = tokenizer.decode(ids, skip_special_tokens=False)
    print("Decoded:", decode)