import re
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, GemmaForCausalLM, LlamaForCausalLM, GenerationConfig
from torch.nn.functional import one_hot
from typing import Dict, Iterable, Optional, Callable, Union

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

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
            self._y, self._y_norm = None, None
            self._a, self._da     = [], {}
            self._h, self._dh     = [], {}

            self._grad_masks = None

        @property
        def attentions(self):
            '''Attention scores of the last batch with shape = (bs, n_layers, n_heads, n_outputs, n_inputs)'''
            if len(self._a) == 0: return None

            # extract dimensions:
            bs, n_layers, n_heads, n_inputs = self._a[-1].shape
            n_outputs = len(self._a)

            # init array:
            attentions = torch.zeros(
                (bs, n_layers, n_heads, n_outputs, n_inputs),
                dtype=float, device=self._a[-1].device, requires_grad=False
            )

            # copy attentions scores:
            for i, a in enumerate(self._a):
                attentions[:, :, :, i, :a.shape[3]] = a

            return attentions

        @property
        def hiddenStates(self):
            '''Hidden states of the last batch with shape = (bs, n_layers, n_outputs, n_inputs, encoding_size)'''
            if len(self._h) == 0: return None

            # extract dimensions:
            bs, n_layers, _, encoding_size = self._h[-1].shape
            n_inputs  = sum(h.shape[2] for h in self._h)
            n_outputs = len(self._h)

            # init array:
            hidden_states = torch.zeros(
                (bs, n_layers, n_outputs, n_inputs, encoding_size),
                dtype=float, device=self._h[-1].device, requires_grad=False
            )

            # Attention Caching: Gradients are actually only computed for last token,
            # but attentions for previous tokens are reused.
            # Workaround: we reuse the gradients calculated for previous tokens at
            # each new output.
            j = 0 
            for i, h in enumerate(self._h):
                k = j + h.shape[2]
                hidden_states[:, :, i:, j:k, :] = h.unsqueeze(dim=2)
                j = k

            return hidden_states

        def approximateOutputProbability(self, target:Iterable[str]):
            '''Approximates the probability of a specific output to happen.

            Args:
                target:     Iterable of tokens for which to compute the probability.

            Returns:        Token probabilities of the last next-token-prediction step with shape = (bs,)'''
  
            if len(target) > self._y_norm.shape[1]:
                return 0.

            p = 1.
            for i, t in enumerate(target):
                p *= self._y_norm[:,i,self._grad_masks[t].to(bool)]

            return p[:,0]

        def getAttentionGradients(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''Jacobian with regard to the attention scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_heads, n_classes, n_outputs, n_inputs)'''

            if len(self._da) == 0: return None, None
            if outputs is None: outputs = self._da

            # extract dimensions:
            bs, _, n_heads, n_inputs = self._a[-1].shape
            n_outputs = len(self._a)
            n_classes = len(outputs)

            # compute jacobian:
            jacobian = torch.zeros(
                (bs, n_heads, n_classes, n_outputs, n_inputs), 
                dtype=float, device=self._a[-1].device, requires_grad=False
            )

            for i, key in enumerate(outputs):
                for j, da in enumerate(self._da[key]):
                    jacobian[:,:,i,j,:da.shape[3]] = da[:,layer,:,:]

            return jacobian

        def getHiddenStateGradients(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''Jacobian with regard to the hidden states of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs, encoding_size)'''

            if len(self._dh) == 0: return None
            if outputs is None: outputs = self._dh

            # extract dimensions:
            bs, _, _, encoding_size = self._h[-1].shape
            n_inputs  = sum(h.shape[2] for h in self._h)
            n_outputs = len(self._h)
            n_classes = len(outputs)

            # compute jacobian:
            jacobian = torch.zeros(
                (bs, n_classes, n_outputs, n_inputs, encoding_size),
                dtype=float, device=self._h[-1].device, requires_grad=False
            )

            # Attention Caching: Gradients are actually only computed for last token,
            # but attentions for previous tokens are reused.
            # Workaround: we reuse the gradients calculated for previous tokens at
            # each new output.
            for i, key in enumerate(outputs):
                k = 0 
                for j, dh in enumerate(self._dh[key]):
                    l = k + dh.shape[2]
                    jacobian[:,i,j:,k:l,:] = dh[:,layer,:,:].unsqueeze(dim=1)
                    k = l

            return jacobian

        def aGrad(self, layer:int=-1, outputs:Optional[Iterable[str]]=None):
            '''AGrad (`-da ⊙ a`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default -1).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_heads, n_classes, n_outputs, n_inputs)'''

            # compute attention weigths and gradients:
            a  = self.attentions[:,layer]
            da = self.getAttentionGradients(layer, outputs)

            return -da * torch.unsqueeze(a, 2)

        def repAGrad(self, outputs:Optional[Iterable[str]]=None):
            '''RepAGrad scores of the last batch.

            Args:
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_heads, n_classes, n_outputs, n_inputs)'''

            raise NotImplementedError()

        def iGrad(self, layer:int=0, alpha:float=1., beta:float=0., outputs:Optional[Iterable[str]]=None):
            '''Inverted gradient scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                alpha:      Multiplicative offset for the decision boundary (default 1.).
                beta:       Additive offset for the decision boundary (default 0.).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs, encoding_size)'''

            if outputs is None: outputs = self._grad_masks

            # compute jacobian:
            dh = self.getHiddenStateGradients(layer, outputs)

            # extract dimensions (bs x n_classes x n_outputs x n_inputs x encoding_size):
            bs, n_classes, n_outputs, n_inputs, encoding_size = dh.shape

            # reshape to (n_classes x (bs x n_outputs x n_inputs x encoding_size)):
            dh = dh.transpose(0,1).flatten(start_dim=1)

            # invert jacobian (shape: bs x n_outputs x n_inputs x encoding_size x n_classes)):
            dh = -torch.linalg.pinv(dh).view((bs, n_outputs, n_inputs, encoding_size, n_classes))

            # get last output (shape: bs x n_outputs x n_classes):
            y = torch.concat([self._y[:,:,self._grad_masks[key].to(bool)] for key in outputs], dim=-1)

            # calculate distance on outputs (shape: bs x n_outputs x n_classes):
            diff_y = y.max(dim=-1, keepdim=True).values - y

            # scale and shift:
            diff_y *= alpha
            diff_y += beta

            # calculate distance on input (shape: bs x n_classes x n_outputs x n_inputs x encoding_size):
            diff_h = torch.zeros((bs, n_classes, n_outputs, n_inputs, encoding_size), device=dh.device, dtype=dh.dtype)
            for i in range(bs):
                for j in range(n_outputs):
                    for c in range(n_classes):
                        diff_h[i, c, j, :, :] = dh[i, j, :, :, c] * diff_y[i, j, c]

            return diff_h

        def gradIn(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''GradIn (`dh ⊙ h`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs, encoding_size)'''

            # compute hidden states and gradients:
            h  = self.hiddenStates[:,layer]
            dh = self.getHiddenStateGradients(layer, outputs)

            return dh * torch.unsqueeze(h, 1)

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

                    y = (outputs.logits[:,-1] @ self._grad_masks[key].to(outputs.logits)).sum()

                    # calculate gradients of output:
                    y.backward(retain_graph = True)

                    # save gradients with regard to attention weights:
                    # Iterable of n_output tensors of shape (bs x n_layers x n_heads x n_inputs)
                    if has_attentions:
                        if key not in self._da: self._da[key] = []
                        self._da[key].append(torch.stack(
                            [a.grad[:,:,-1,:].detach().clone().cpu() for a in outputs.attentions],
                            dim=1 
                        ))

                    # save gradients with regard to hidden states:
                    # Iterable of n_output tensors of shape (bs x n_layers x n_inputs x encoding_size)
                    if has_hidden_states:
                        if key not in self._dh: self._dh[key] = []
                        self._dh[key].append(torch.stack(
                            [h.grad.detach().clone().cpu() for h in outputs.hidden_states],
                            dim=1 
                        ))

            # save attentions:
            # Iterable of n_output tensors of shape (bs x n_layers x n_heads x n_inputs)
            if has_attentions:
                self._a.append(torch.stack(
                    [a[:,:,-1,:].detach().clone().cpu() for a in outputs.attentions],
                    dim=1 
                ))

            # save hidden states:
            # Iterable of n_output tensors of shape (bs x n_layers x n_inputs x encoding_size)
            if has_hidden_states:
                self._h.append(torch.stack(
                    [h.detach().clone().cpu() for h in outputs.hidden_states],
                    dim=1 
                ))

            # save last sequence:
            if self._y is None:
                self._y      = outputs.logits.detach().cpu()[:,-1:,:]
                self._y_norm = torch.nn.functional.softmax(outputs.logits.detach().cpu()[:,-1:,:], dim=-1)

            else:
                self._y      = torch.concat([self._y, outputs.logits.detach().cpu()[:,-1:,:]], dim=1)
                self._y_norm = torch.concat([self._y_norm, torch.nn.functional.softmax(outputs.logits.detach().cpu()[:,-1:,:], dim=-1)], dim=1)

            # reset gradient computation:
            if self._grad_masks is not None: torch.set_grad_enabled(prev_grad)

            return outputs

        def generate(self, input_ids:torch.Tensor, outputs:Optional[Dict[str, int]]=None, output_size:Optional[int]=None, **kwargs):
            ''' Generates sequences of token ids for models with a language modeling head and activates gradient computation for specific outputs so that their corresponding explanations can be computed.

            Args:
                input_ids:      The sequence used as a prompt for the generation or as model inputs to the encoder.
                outputs:        Dictionary of output names and indices to compute the scores for (optional).
                outputs_size:   Output size of the model (optional).
                **kwargs:       Any keyword arguments accepted by [`GenerationMixin.generate(...)`](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1884) (optional).
            
            Returns:
                [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
                or when `config.return_dict_in_generate=True`) or a `torch.LongTensor`.

                    If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                    [`~utils.ModelOutput`] types are:

                        - [`~generation.GenerateDecoderOnlyOutput`],
                        - [`~generation.GenerateBeamDecoderOnlyOutput`]

                    If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                    [`~utils.ModelOutput`] types are:

                        - [`~generation.GenerateEncoderDecoderOutput`],
                        - [`~generation.GenerateBeamEncoderDecoderOutput`]
            '''
            
            # register output masks for gradient computation:
            if outputs is not None:
                # Forward must have been called once at least (usually during training):
                if output_size is None:
                    if self._y is None:
                        raise Exception('GenericExplainableCausalModel.forward(...) must have been called at least once (usually during training) before calling GenericExplainableModel.registerOutputsForExplanation(...).')

                    else:
                        output_size = self._y.shape[-1]

                # Build masks:
                self._grad_masks = {
                    key:one_hot(torch.tensor(outputs[key]), output_size)
                    for key in outputs
                }

            else: self._grad_masks = {}

            # reset state:
            self._y, self._y_norm = None, None
            self._a, self._da     = [], {}
            self._h, self._dh     = [], {}

            # generate outputs:
            return super().generate(inputs=input_ids, **kwargs)

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

        # cached variables:
        self._token_cache = []

    def _outputs2set(self, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]):
        # use cached values (set in call to `self.generate(...)`) if outputs is None:
        if outputs is None: return self._token_cache

        # otherwise compute values:
        tokens, token_indices = [], []

        for output in outputs: 
            # convert strings to Iterable of tokens:
            if isinstance(output, str):
                output = self.tokenizer.convert_ids_to_tokens(self.tokenizer(output)['input_ids'][1:])

            # add new tokens to set and track their indices:
            token_indices.append([])
            for token in output:
                try: token_indices[-1].append(tokens.index(token))
                except ValueError:
                    token_indices[-1].append(len(tokens))
                    tokens.append(token)

        return tokens, token_indices

    @property
    def mask_token_id(self):
        return self.tokenizer.unk_token_id

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, max_seq_len:int, max_gen_len:int, **kwargs):
        prefix = pretrained_model_name_or_path.split('/')[0].lower()
        if prefix == "meta-llama":  return LlamaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        if prefix == "google":      return GemmaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        if prefix == "mistralai":   return MistralChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        raise NotImplementedError()

    def approximateOutputProbability(self, target:Union[str,Iterable[str]]):
        '''Approximates the probability of a specific output to happen.

        Args:
            target:     String or Iterable of tokens for which to compute the probability.

        Returns:        Token probabilities of the last next-token-prediction step with shape = (bs,)'''
        # convert string to Iterable of tokens:
        if isinstance(target, str):
            target = self.tokenizer.convert_ids_to_tokens(self.tokenizer(target)['input_ids'][1:])

        # call base method:
        return self.model.approximateOutputProbability(list(target) + [])

    def grad(self, layer:int=-1, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None):
        '''Jacobian with regard to the hidden states of the last batch.

        Args:
            layer:      Transformer layer to compute the scores for (default 0).
            outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).

        Returns:        Importance scores with shape = (bs, n_outputs, n_inputs, encoding_size)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores:
        grad = self.model.getHiddenStateGradients(layer=layer, outputs=tokens)

        # aggregate per label:
        n_outputs = grad.shape[2]
        grad = torch.stack([
            torch.stack([
                grad[0, i, j, :] for j, i in enumerate(indices[:n_outputs])
            ], dim=0).cumsum(dim=0).mean(dim=0) for indices in token_indices],
            dim=0
        )

        return grad

    def aGrad(self, layer:int=-1, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None):
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            layer:      Transformer layer to compute the scores for (default -1).
            outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).

        Returns:        Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores:
        aGrad = self.model.aGrad(layer=layer, outputs=tokens)

        # aggregate per label:
        n_outputs = aGrad.shape[3]
        aGrad = torch.stack([
            torch.stack([
                aGrad[0, :, i, j, :] for j, i in enumerate(indices[:n_outputs])
            ], dim=0).cumsum(dim=0).mean(dim=0) for indices in token_indices],
            dim=0
        )

        return aGrad

    def repAGrad(self, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None):
        '''RepAGrad scores of the last batch.

        Args:
            outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).

        Returns:        Importance scores with shape = (bs, n_heads, n_outputs, n_inputs)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores:
        repAGrad = self.model.repAGrad(outputs=tokens)

        # aggregate per label:
        n_outputs = repAGrad.shape[3]
        repAGrad = torch.stack([
            torch.stack([
                repAGrad[0, :, i, j, :] for j, i in enumerate(indices[:n_outputs])
            ], dim=0).cumsum(dim=0).mean(dim=0) for indices in token_indices],
            dim=0
        )

        return repAGrad

    def iGrad(self, layer:int=0, alpha:float=1., beta:float=0., outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None):
        '''Inverted gradient scores of the last batch.

        Args:
            alpha:      Multiplicative offset for the decision boundary.
            beta:       Additive offset for the decision boundary.
            layer:      Transformer layer to compute the scores for (default 0).
            outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).

        Returns:        Importance scores with shape = (bs, n_outputs, n_inputs, encoding_size)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores:
        iGrad = self.model.iGrad(layer=layer, alpha=alpha, beta=beta, outputs=tokens)

        # aggregate per label:
        n_outputs = iGrad.shape[2]
        iGrad = torch.stack([
            torch.stack([
                iGrad[0, i, j, :] for j, i in enumerate(indices[:n_outputs])
            ], dim=0).cumsum(dim=0).mean(dim=0) for indices in token_indices],
            dim=0
        )

        return iGrad

    def gradIn(self, layer:int=0, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None):
        '''GradIn (`dh ⊙ h`) scores of the last batch.

        Args:
            layer:      Transformer layer to compute the scores for (default 0).
            outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).

        Returns:        Importance scores with shape = (bs, n_outputs, n_inputs, encoding_size)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores:
        gradIn = self.model.gradIn(layer=layer, outputs=tokens)

        # aggregate per label:
        n_outputs = gradIn.shape[2]
        gradIn = torch.stack([
            torch.stack([
                gradIn[0, i, j, :] for j, i in enumerate(indices[:n_outputs])
            ], dim=0).cumsum(dim=0).mean(dim=0) for indices in token_indices],
            dim=0
        )

        return gradIn

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

    def generate(self, txt:str, history:Optional[torch.Tensor]=None, compute_grads:Union[Iterable[str],Iterable[Iterable[str]]]=[], **kwargs):
        # format prompt:
        prompt = self.getPrompt(txt)
        if history is not None: prompt = self._eot + prompt

        # tokenize:
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.model.device)
        if history is not None: input_ids = torch.concatenate((history[:,:-1], input_ids[:,1:]), axis=-1)

        # generate and register gradient computation:
        kwargs.update(self.model_kwargs)
        self._token_cache = self._outputs2set(compute_grads)
        output_ids = self.model.generate(
            input_ids=input_ids, 
            outputs={token: self.tokenizer.vocab[token] for token in self._token_cache[0]},
            output_size=len(self.tokenizer.vocab),
            **kwargs
        )

        #return:
        return self._rex.findall(self.tokenizer.decode(output_ids[0,:-1]) + self._eot), input_ids, output_ids


class GemmaChatGenerationPipeline(ChatGenerationPipeline):
    def __init__(self, tokenizer:PreTrainedTokenizer, model:PreTrainedModel, max_seq_len:int, max_gen_len:int):
        super().__init__(
            tokenizer = tokenizer,
            model     = model,
            bos       = '<bos>',
            sot       = lambda role: f'<start_of_turn>{role}\n',
            eot       = '<end_of_turn>\n'
        )

        self.model.generation_config = GenerationConfig(
            bos_token_id = tokenizer.bos_token_id, #2
            eos_token_id = tokenizer.eos_token_id, #1
            pad_token_id = tokenizer.pad_token_id, #0
            
            max_length = max_seq_len,
            max_new_tokens = max_gen_len,
        )

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, max_seq_len:int, max_gen_len:int, **kwargs):
        return GemmaChatGenerationPipeline(
            tokenizer   = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model       = GenericModelForExplainableCausalLM(GemmaForCausalLM).from_pretrained(pretrained_model_name_or_path, **kwargs),
            max_seq_len = max_seq_len,
            max_gen_len = max_gen_len
        )

    def getPrompt(self, txt:str, bos:bool=False):
        # create prompt string:
        prompt = self._sot('user') + txt + self._eot + self._sot('model')

        # add bos token:
        if bos: prompt = self._bos + prompt

        return prompt


class LlamaChatGenerationPipeline(ChatGenerationPipeline):
    def __init__(self, tokenizer:PreTrainedTokenizer, model:PreTrainedModel, max_seq_len:int, max_gen_len:int, pad_token_id:Optional[int]=None):
        super().__init__(
            tokenizer    = tokenizer,
            model        = model,
            bos          = '<|begin_of_text|>',
            sot          = lambda role: f'<|start_header_id|>{role}<|end_header_id|>\n\n',
            eot          = '<|eot_id|>',
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            pad_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>') if pad_token_id is None else pad_token_id
        )

        self.model.generation_config = GenerationConfig(
            bos_token_id = tokenizer.convert_tokens_to_ids('<|begin_of_text|>'),
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            pad_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>') if pad_token_id is None else pad_token_id,
            
            max_length = max_seq_len,
            max_new_tokens = max_gen_len,
        )

    @property
    def mask_token_id(self):
        return self.tokenizer.convert_tokens_to_ids('###')
        #return self.tokenizer.convert_tokens_to_ids('<|reserved_special_token_250|>')

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, max_seq_len:int, max_gen_len:int, pad_token_id:Optional[int]=None, **kwargs):
        return LlamaChatGenerationPipeline(
            tokenizer    = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model        = GenericModelForExplainableCausalLM(LlamaForCausalLM).from_pretrained(pretrained_model_name_or_path, **kwargs),
            max_seq_len  = max_seq_len,
            max_gen_len  = max_gen_len,
            pad_token_id = pad_token_id
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