import re
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, GemmaForCausalLM, LlamaForCausalLM
from typing import List, Dict, Iterable, Optional, Callable

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

def zero_grads(n:torch.Node) -> None:
    if torch.is_tensor(n):
        if n.grad is not None:
            n.grad = None
            zero_grads(n.grad_fn)
        return

    if hasattr(n, 'variable'):
        zero_grads(n.variable)

    if hasattr(n, 'next_functions'):
        for fn in n.next_functions:
            if fn[0] is not None:
                zero_grads(fn[0])

#=======================================================================#
# Generic Model Class:                                                  #
#=======================================================================#

def GenericModelForExplainableCausalLM(T:type):
    # make sure T is derived from PreTrainedModel:
    assert issubclass(T, PreTrainedModel)

    # generic class definition:
    class _GenericModelForExplainableCausalLM(T):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._a, self._da  = [], {}
            self._h, self._dh  = [], {}

            self._grad_masks = None

        @property
        def attentions(self):
            '''Attention scores of the last batch with shape = (bs, n_layers, n_heads, n_inputs, n_inputs)'''
            if len(self._a) == 0: return None

            # extract dimensions:
            bs, n_layers, n_heads, _, n_inputs = self._a[-1].shape

            # init array:
            attentions = torch.zeros(
                (bs, n_layers, n_heads, n_inputs, n_inputs),
                dtype=float, device=self._a[-1].device, requires_grad=False
            )

            # copy attentions scores:
            i = 0
            for a in self._a:
                j = i + a.shape[3]
                attentions[:, :, :, i:j, :j] = a
                i = j

            return attentions

        @property
        def hiddenStates(self):
            '''Hidden states of the last batch with shape = (bs, n_layers, n_inputs, encoding_size)'''
            if len(self._h) == 0: return None

            # extract dimensions:
            bs, n_layers, _, encoding_size = self._h[-1].shape
            n_inputs = sum([h.shape[2] for h in self._h])

            # init array:
            hidden_states = torch.zeros(
                (bs, n_layers, n_inputs, encoding_size),
                dtype=float, device=self._h[-1].device, requires_grad=False
            )

            # copy attentions scores:
            i = 0
            for h in self._h:
                j = i + h.shape[2]
                hidden_states[:, :, i:j, :] = h
                i = j

            return hidden_states

        def getAttentionGradients(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''Jacobian with regard to the attention scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_outputs, n_heads, n_inputs, n_inputs)'''

            if len(self._da) == 0: return None, None
            if outputs is None: outputs = self._da

            # extract dimensions:
            bs, _, n_heads, _, n_inputs = self._a[-1].shape

            # compute jacobian:
            jacobian = torch.zeros(
                (bs, len(outputs), n_heads, n_inputs, n_inputs), 
                dtype=float, device=self._a[-1].device, requires_grad=False
            )

            for i, key in enumerate(outputs):
                j = 0
                for da in self._da[key]:
                    k = j + da.shape[3]
                    jacobian[:,i,:,j:k,:k] = da[:,layer,:,:,:]
                    j = k

            return jacobian

        def getHiddenStateGradients(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''Jacobian with regard to the hidden states of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_outputs, n_inputs, encoding_size)'''

            if len(self._dh) == 0: return None
            if outputs is None: outputs = self._dh

            # extract dimensions:
            bs, _, _, encoding_size = self._h[-1].shape
            n_inputs = sum([h.shape[2] for h in self._h])

            # compute jacobian:
            jacobian = torch.zeros(
                (bs, len(outputs), n_inputs, encoding_size),
                dtype=float, device=self._h[-1].device, requires_grad=False
            )

            for i, key in enumerate(outputs):
                j = 0
                for dh in self._dh[key]:
                    k = j + dh.shape[2]
                    jacobian[:, i, j:k,:] = dh[:,layer,:,:]
                    j = k

            return jacobian

        def aGrad(self, layer:int=-1, outputs:Optional[Iterable[str]]=None):
            '''AGrad (`-da ⊙ a`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default -1).
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_outputs, n_heads, n_inputs, n_inputs)'''

            # compute attention weigths and gradients:
            a  = self.attentions[:,layer]
            da = self.getAttentionGradients(layer, outputs)

            return -da * torch.unsqueeze(a, 1)

        def repAGrad(self, outputs:Optional[Iterable[str]]=None):
            '''RepAGrad scores of the last batch.

            Args:
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (n_outputs, bs, n_heads, n_inputs, n_inputs)'''

            raise NotImplementedError()

        def iGrad(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''Inverted gradient scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_inputs, encoding_size, n_outputs)'''

            # compute jacobian:
            jacobian = self.getHiddenStateGradients(layer, outputs)

            # extract dimensions:
            bs, n_outputs, n_inputs, encoding_size = jacobian.shape

            # reshape to (n_outputs x ...):
            j = jacobian.transpose(0,1).flatten(start_dim=1)

            # invert jacobian:
            return -torch.linalg.pinv(j).reshape((bs, n_inputs, encoding_size, n_outputs))

        def hGrad(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''`dh · h` scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_outputs, n_inputs)
            '''

            # compute hidden states and gradients:
            h  = self.hiddenStates[:,layer]
            dh = self.getHiddenStateGradients(layer, outputs)

            # extract dimensions:
            bs, n_outputs, n_inputs, _ = dh.shape

            # compute dot products:
            hgrad = torch.zeros(
                (bs, n_outputs, n_inputs),
                dtype=float, device=h.device, requires_grad=False
            )
            for i in range(bs):
                for j in range(n_inputs):
                    torch.matmul(dh[i, :, :j+1, :].flatten(start_dim=1), h[i, :j+1, :].flatten(), out=hgrad[i, :, j]) 

            return hgrad

        def GradIn(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''GradIn (`dh ⊙ h`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    List of output names to comput the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_outputs, n_inputs, encoding_size)'''

            # compute hidden states and gradients:
            h  = self.hiddenStates[:,layer]
            dh = self.getHiddenStateGradients(layer, outputs)

            return dh * torch.unsqueeze(h, 1)

        def registerGradientComputation(self, masks=Dict[str,torch.Tensor]):
            self._grad_masks = {key:masks[key].to(self.device) for key in masks}

        def forward(self, input_ids:torch.Tensor, *args, **kwargs):
            # control gradient computation:
            prev_grad = torch.is_grad_enabled()
            if self._grad_masks is not None: torch.set_grad_enabled(True)
            
            # propagate through model:
            outputs = super().forward(input_ids, *args, **kwargs)

            has_attentions    = checkattr(outputs, 'attentions')
            has_hidden_states = checkattr(outputs, 'hidden_states')

            # save gradients:
            if self._grad_masks is not None:
                for key in self._grad_masks:
                    # register attentions for gradient computation:
                    if has_attentions:
                        for a in outputs.attentions:
                            a.retain_grad()
                            a.grad = None

                    # register hidden states for gradient computation:
                    if has_hidden_states:
                        for h in outputs.hidden_states:
                            h.retain_grad()
                            h.grad = None

                    y = outputs.logits[:,-1,self._grad_masks[key].to(outputs.logits.device)].sum()

                    # reset gradients:
                    #zero_grads(y)

                    # calculate gradients of output:
                    y.backward(retain_graph = True)

                    # save gradients with regard to attention weights:
                    if has_attentions:
                        if key not in self._da: self._da[key] = []
                        self._da[key].append(torch.concatenate([
                            torch.reshape(
                                a.grad.detach().clone().cpu(), 
                                (a.shape[0], 1) + a.shape[1:]
                            ) for a in outputs.attentions],
                            dim=1 
                        ))

                    # save gradients with regard to hidden states:
                    if has_hidden_states:
                        if key not in self._dh: self._dh[key] = []
                        self._dh[key].append(torch.concatenate([
                            torch.reshape(
                                h.grad.detach().clone().cpu(), 
                                (h.shape[0], 1) + h.shape[1:]
                            ) for h in outputs.hidden_states],
                            dim=1 
                        ))

            # save attentions:
            if has_attentions:
                self._a.append(torch.concatenate([
                    torch.reshape(
                        atts.detach().cpu(), 
                        (atts.shape[0], 1) + atts.shape[1:]
                    ) for atts in outputs.attentions],
                    dim=1 
                ))

            # save hidden states:
            if has_hidden_states:
                self._h.append(torch.concatenate([
                    torch.reshape(
                        h.detach().cpu(), 
                        (h.shape[0], 1) + h.shape[1:]
                    ) for h in outputs.hidden_states],
                    dim=1 
                ))

            # reset gradient computation:
            if self._grad_masks is not None: torch.set_grad_enabled(prev_grad)

            return outputs

        def generate(self, *args, **kwargs):
            # reset state:
            self._a, self._da  = [], {}
            self._h, self._dh  = [], {}

            # generate outputs:
            outputs = super().generate(*args, **kwargs)

            # reset gradient computation:
            self._grad_masks = None

            return outputs

    # return generic class subclassed from T:
    return _GenericModelForExplainableCausalLM

#=======================================================================#
# Generation Pipelines:                                                 #
#=======================================================================#

class ChatGenerationPipeline:
    def __init__(self, tokenizer:PreTrainedTokenizer, model:PreTrainedModel, bos:str, sot:Callable[[str],str], eot:str, **kwargs):
        # pipeline components:
        self.tokenizer    = tokenizer
        self.model        = model
        self.model_kwargs = kwargs
        
        # chat format:
        self._bos = bos
        self._sot = sot
        self._eot = eot

        rex = self._sot('##role##') + "##content##" + self._eot
        rex = rex.replace('(', '\\(')
        rex = rex.replace(')', '\\)')
        rex = rex.replace('<', '\\<')
        rex = rex.replace('>', '\\>')
        rex = rex.replace('|', '\\|')
        rex = rex.replace('##role##', '(?P<role>\w*?)')
        rex = rex.replace('##content##', '(?P<content>.*?)')
        self._rex = re.compile(rex, re.DOTALL)

    @property
    def mask_token_id(self):
        return self.tokenizer.unk_token_id

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, **kwargs):
        prefix = pretrained_model_name_or_path.split('/')[0].lower()
        if prefix == "meta-llama":  return LlamaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if prefix == "google":      return GemmaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if prefix == "mistralai":   return MistralChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)
        raise NotImplementedError()

    def countTokens(self, txt:str, role:str='user', sot:bool=False, eot:bool=False):
        s = ''

        # include start of turn:
        if sot: s += self._bos + self._sot(role)

        # add text:
        s += txt

        # include end of turn:
        if eot: s += self._eot + self._sot(role)

        print(self.tokenizer.convert_ids_to_tokens(self.tokenizer(s, add_special_tokens=False)['input_ids']))
        return len(self.tokenizer(s, add_special_tokens=False)['input_ids'])
    
    def getPostfixSize(self, txt:str):
        return len(self.tokenizer(txt + self._eot)['input_ids'])
    
    def getPrompt(self, txt:str, bos:bool=False):
        raise NotImplementedError()

    def generate(self, txt:str, history:Optional[torch.Tensor]=None, compute_grads:Optional[Iterable[str]]=None, **kwargs):
        # format prompt:
        prompt = self.getPrompt(txt)
        if history is not None: prompt = self._eot + prompt

        # tokenize:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        if history is not None: input_ids = torch.concatenate((history[:,:-1], input_ids[:,1:]), axis=-1)

        # register gradient computation:
        if compute_grads is not None:
            self.model.registerGradientComputation({
                key: torch.tensor([t==key for t in self.tokenizer.vocab], device='cuda')
                for key in compute_grads
            })

        # generate:
        kwargs.update(self.model_kwargs)
        output_ids = self.model.generate(input_ids=input_ids, **kwargs)

        #return:
        return self._rex.findall(self.tokenizer.decode(output_ids[0,:-1]) + self._eot), input_ids, output_ids


class GemmaChatGenerationPipeline(ChatGenerationPipeline):
    def __init__(self, tokenizer:PreTrainedTokenizer, model:PreTrainedModel):
        super().__init__(
            tokenizer = tokenizer,
            model     = model,
            bos       = '<bos>',
            sot       = lambda role: f'<start_of_turn>{role}\n',
            eot       = '<end_of_turn>\n'
        )

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, **kwargs):
        return GemmaChatGenerationPipeline(
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model     = GenericModelForExplainableCausalLM(GemmaForCausalLM).from_pretrained(pretrained_model_name_or_path, **kwargs)
        )

    def getPrompt(self, txt:str, bos:bool=False):
        # create prompt string:
        prompt = self._sot('user') + txt + self._eot + self._sot('model')

        # add bos token:
        if bos: prompt = self._bos + prompt

        return prompt


class LlamaChatGenerationPipeline(ChatGenerationPipeline):
    def __init__(self, tokenizer:PreTrainedTokenizer, model:PreTrainedModel, pad_token_id:Optional[int]=None):
        super().__init__(
            tokenizer    = tokenizer,
            model        = model,
            bos          = '<|begin_of_text|>',
            sot          = lambda role: f'<|start_header_id|>{role}<|end_header_id|>\n\n',
            eot          = '<|eot_id|>',
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            pad_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>') if pad_token_id is None else pad_token_id
        )

    @property
    def mask_token_id(self):
        return self.tokenizer.convert_tokens_to_ids('###')
        #return self.tokenizer.convert_tokens_to_ids('<|reserved_special_token_250|>')

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, **kwargs):
        return LlamaChatGenerationPipeline(
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model     = GenericModelForExplainableCausalLM(LlamaForCausalLM).from_pretrained(pretrained_model_name_or_path, **kwargs)
        )

    def getPrompt(self, txt:str, bos:bool=False):
        # create prompt string:
        prompt = self._sot('user') + txt + self._eot + self._sot('assistant')

        # add bos token:
        if bos: prompt = self._bos + prompt

        return prompt


class MistralChatGenerationPipeline(ChatGenerationPipeline):
    def __init__(self, tokenizer:PreTrainedTokenizer, model:PreTrainedModel, pad_token_id:Optional[int]=None):
        super().__init__(
            tokenizer    = tokenizer,
            model        = model,
            bos          = '<s>',
            sot          = lambda role: f'[/INST] ',
            eot          = ' [/INST]',
        )

    @property
    def mask_token_id(self):
        return self.tokenizer.convert_tokens_to_ids('<unk>')

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, **kwargs):
        return LlamaChatGenerationPipeline(
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model     = GenericModelForExplainableCausalLM(LlamaForCausalLM).from_pretrained(pretrained_model_name_or_path, **kwargs)
        )