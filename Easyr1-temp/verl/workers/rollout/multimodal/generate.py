from ..multimodal.warp_output import text2vllm_like_output, single_output
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import torch
from collections import defaultdict

class StopOnTokenSequence(StoppingCriteria):
    def __init__(self, input_len, tokenizer):
        """
        stop_sequences: list of tokenized stop word sequences (each is a list of token IDs)
        """
        #self.stop_dict = stop_dict
        #self.stop_exc_dict = stop_exc_dict
        #self.stop_words = stop_words
        self.input_len = input_len
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """
        check if the generated texts match the stop words
        """
        assert input_ids.shape[0] == 1 # only support batch_size == 1
        #if input_ids.shape[1]<self.input_len + 10:
        #    return False
        generated_text = self.tokenizer.decode(input_ids[0][self.input_len:])
        if generated_text[-6:]=="Step 1":
            return False
        if generated_text[-6:-2] == "Step":
            return True
        
        '''for stop_word, stop_ids in self.stop_dict.items():
                flag =True
                for stop_exc_ids in self.stop_exc_dict[stop_word]:    
                    cond = (input_ids.shape[1] >= stop_ids.shape[1] and\
                        torch.equal(input_ids[0][-stop_exc_ids.shape[1]:-stop_exc_ids.shape[1]+stop_ids.shape[1]], stop_ids[0])\
                        and not torch.equal(input_ids[0][-stop_exc_ids.shape[1]:], stop_exc_ids[0]))
                    flag = flag and cond
                if flag: # once for a stop word, all stop_exc pass
                    return True # True (stop generation) only when all `cond` is true'''
        return False

def get_stopping_criteria(mllm_processor, input_ids_len):
    #stop_words = config.stop
    '''stop_exception_words = config.stop_exception
    stop_dict = {}
    stop_exc_dict = defaultdict(list)
    for word in stop_words:
        stop_dict[word] = mllm_processor.tokenizer.encode(word, add_special_tokens=False, return_tensors="pt").to(int(config.allowed_devices[0]))
        for word, exc_word_list in stop_exception_words.items():
            for exc_word in exc_word_list:
                stop_exc_dict[word].append(mllm_processor.tokenizer.encode(exc_word, add_special_tokens=False, return_tensors="pt").to(int(config.allowed_devices[0])))'''
                
    #stopping_criteria = StoppingCriteriaList([StopOnTokenSequence(stop_dict , stop_exc_dict, input_ids_len, mllm_processor.tokenizer)])
    stopping_criteria = StoppingCriteriaList([StopOnTokenSequence(input_ids_len, mllm_processor.tokenizer)])
    return stopping_criteria


def create_hook_fn(last_layer_attn: list):
    def hook(module, input, output):
        # output[0] is the hidden state, output[1] is the attention
        # output[0].shape # torch.Size([1, 1602, 3584]) torch.Size([1, 1, 3584]) torch.Size([1, 1, 3584]) ...
        last_layer_attn.append(output[0])
        return output
    return hook

def generate_multimodal_responses(mllm, mllm_processor, config, multimodal_processed_inputs):

    output_classes = []
    #stopping_criteria = get_stopping_criteria(mllm_processor, multimodal_processed_inputs.input_ids.shape[1])
    multimodal_processed_inputs = multimodal_processed_inputs.to(int(config.allowed_devices[0]))
    # sample `config.best_of`` times
    
    for response_id in range(config.n_generate_sample):
        print(f"---------- Generating response {response_id}/{config.n_generate_sample} ----------")
        
        if "Qwen" in config.mllm_dir or "ThinkLite-VL" in config.mllm_dir:
            last_layer_attn_list = []
            hook_fn = create_hook_fn(last_layer_attn_list)
            handle = mllm.model.layers[-1].self_attn.register_forward_hook(hook_fn)
        else:
            raise NotImplementedError
        
        return_dict = False
        if not config.do_sample:
            torch.manual_seed(42)
        generated_ids = mllm.generate(**multimodal_processed_inputs, 
                                                        temperature=config.temperature, # 0.7
                                                        top_k=100,
                                                        do_sample=config.do_sample,
                                                        #top_p=config.top_p,
                                                        #num_return_sequences=1, # if>1, OOM
                                                        max_new_tokens=config.max_tokens, 
                                                        #stop=config.stop,
                                                        #skip_special_tokens=False,
                                                        #stopping_criteria=stopping_criteria, # stop generation when outputting "<end_of_step>", "<end_of_output>", or "<end_of_answer>"
                                                        #output_attentions=True,
                                                        #output_hidden_states=True
                                                        output_scores=False,
                                                        return_dict_in_generate=return_dict
                                                        )
        if return_dict:
            output_ids = generated_ids.sequences
        else:
            output_ids = generated_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip( multimodal_processed_inputs.input_ids,  output_ids)
        ]
        output_texts = mllm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ) # all output text of a batch

        # for each output_class (a question) in the batch, add a response generated by the model
        if len(output_classes) == 0:
            for output_text, generated_id in zip(output_texts, generated_ids_trimmed):
                output_class = text2vllm_like_output()
                output = single_output(index=response_id, text=output_text, token_ids=tuple(generated_id.tolist()))
                output_class.add_a_response(output)
                output_classes.append(output_class)
        else:
            for output_class, output_text, generated_id in zip(output_classes,output_texts,generated_ids_trimmed):
                output = single_output(index=response_id, text=output_text, token_ids=tuple(generated_id.tolist()))
                output_class.add_a_response(output)
                
        handle.remove()

    del multimodal_processed_inputs
    return output_classes