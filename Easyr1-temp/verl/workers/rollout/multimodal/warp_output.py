class text2vllm_like_output:
    def __init__(self):
        self.outputs = []
    
    def add_a_response(self, response=None):
        self.outputs.append(response)
    
class single_output:
    def __init__(self, index=None, text=None, token_ids=None, cumulative_logprob=None, logprobs=None, finish_reason=None, stop_reason=None):
        self.index = index
        self.text = text
        self.token_ids = token_ids
        self.cumulative_logprob = cumulative_logprob
        self.logprobs = logprobs
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason