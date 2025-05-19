import re
import shap
import torch
import torch.types
import numpy as np
from transformers import PreTrainedTokenizer, PreTrainedModel, AutoTokenizer, GemmaForCausalLM, Gemma2ForCausalLM, LlamaForCausalLM, GenerationConfig
from torch.nn.functional import one_hot
from typing import List, Dict, Iterable, Optional, Callable, Union, Literal

#=======================================================================#
# Helper Functions:                                                     #
#=======================================================================#

class DistributedStackedTensor:
    def __init__(self, shards:List[torch.Tensor], dim:int, devices:Optional[List[torch.device]]=None, dtype:Optional[type]=None):
        assert dim < shards[0].dim()

        self.shards = shards
        self.dim = torch.tensor(dim, dtype=int)

        if dtype is None: dtype = shards[0].dtype
        if devices is None: devices = [shard.device for shard in shards]

        for i, device in enumerate(devices):
            self.shards[i] = self.shards[i].to(device=device, dtype=dtype)

        self.limits = torch.concat(
            (torch.tensor([0,], dtype=int),
             torch.cumsum(torch.tensor([shard.shape[self.dim] for shard in self.shards], dtype=int), dim=0)),
            dim=0
        )

    def __indices2shards(self, indices:Iterable[int]):
        indices_shards = [[]]
        j = 1
        for i in indices:
            while i >= self.limits[j]:
                indices_shards.append([])
                j += 1

            indices_shards[-1].append(i-self.limits[j-1])

        return [torch.tensor(indices_shard) for indices_shard in indices_shards]

    def __getitem__(self, indices):
        # convert to tuple:
        if not isinstance(indices, tuple): indices = (indices,)

        if len(indices) <= self.dim: return torch.concat(
            [shard.__getitem__(indices) for shard in self.shards],
            dim = self.dim - sum([isinstance(i, int) for i in indices])
        )

        key = indices[self.dim]

        if torch.is_tensor(key):
            if key.dim() == 0:
                key = int(key.detach().cpu())

        if isinstance(key, int):
            i = torch.sum(self.limits[1:] <= key)
            j = key - self.limits[i]

            return self.shards[i].__getitem__(indices[:self.dim] + (j,) + indices[self.dim + 1:])
        
        if isinstance(key, slice):
            start = key.start
            if start is None:   start = 0

            step = key.step
            if step is None:    step = 1

            stop = key.stop
            if stop is None:    stop = self.limits[-1]
            elif stop < 0:      stop = self.limits[-1] + stop

            dim = self.dim - sum([isinstance(i, int) for i in indices[:self.dim]])
            indices_fixed = indices[:self.dim] + (slice(None),) + indices[self.dim + 1:]

            shards = []
            for shard, indices_shard in zip(self.shards, self.__indices2shards(range(start, stop, step))):
                if len(indices_shard) > 0:
                    #print(shard.shape, tuple([slice(None),]*dim) + (indices_shard,))
                    shard = shard.__getitem__(tuple([slice(None),]*dim) + (indices_shard,))
                    #print(shard.shape, indices_fixed)
                    shard = shard.__getitem__(indices_fixed)
                    #print(shard.shape)
                    shards.append(shard)

            return torch.concatenate(shards, dim=dim)

        if isinstance(key, Iterable):
            dims_iter, dims_int = [], []
            for d, index in enumerate(indices):
                if isinstance(index, Iterable): dims_iter.append(d)
                if isinstance(index, int):      dims_int.append(d)

            dim = min(dims_iter)
            dim -= sum([d < dim for d in dims_int])

            shards = []
            i = 0
            for shard, indices_shard in zip(self.shards, self.__indices2shards(key)):
                if len(indices_shard) > 0:
                    n = len(indices_shard)

                    indices_shard = []
                    for d, index in enumerate(indices):
                        indices_shard.append(index[i:i+n] if d in dims_iter else index)

                    shard = shard.__getitem__(indices_shard)
                    shards.append(shard)

            return torch.concatenate(shards, dim=dim)
        
        raise KeyError(indices)

    def _apply_reduction(self, reduce:Callable[[torch.Tensor],torch.Tensor], combine:Callable[[torch.Tensor,torch.Tensor],torch.Tensor], dim:Optional[int]=None, keepdim:bool=True):
        result = None
        for shard in self.shards:
            value  = reduce(shard)
            if result is None: result = value
            elif dim is None: combine(result, value)
            elif dim == self.dim: combine(result, value)
            else:
                concat_dim = self.dim
                if dim < self.dim and dim >= 0 and not keepdim: concat_dim -= 1
                result = torch.concatenate((result, value), dim=concat_dim)
        return result

    @property
    def dtype(self):
        return self.shards[0].dtype

    @property
    def shape(self):
        s = self.shards[0].shape
        n = int(self.limits[-1])

        return s[:self.dim] + (n,) + s[self.dim + 1:]

    def absmax(self, dim:Optional[int]=None, keepdim:bool=False):
        return self._apply_reduction(
            lambda x: x.abs().max(dim, keepdim)[0],
            torch.maximum,
            dim, keepdim
        )
    
    def max(self, dim:Optional[int]=None, keepdim:bool=False):
        return self._apply_reduction(
            lambda x: x.max(dim, keepdim)[0],
            torch.maximum,
            dim, keepdim
        )

    def absmin(self, dim:Optional[int]=None, keepdim:bool=False):
        return self._apply_reduction(
            lambda x: x.abs().min(dim, keepdim)[0],
            torch.minimum,
            dim, keepdim
        )

    def min(self, dim:Optional[int]=None, keepdim:bool=False):
        return self._apply_reduction(
            lambda x: x.min(dim, keepdim)[0],
            torch.minimum,
            dim, keepdim
        )

    def abssum(self, dim:Optional[int]=None, keepdim:bool=False):
        return self._apply_reduction(
            lambda x: x.abs().sum(dim, keepdim),
            lambda a,b: a+b,
            dim, keepdim
        )

    def sum(self, dim:Optional[int]=None, keepdim:bool=False):
        return self._apply_reduction(
            lambda x: x.sum(dim, keepdim),
            lambda a,b: a+b,
            dim, keepdim
        )
    
    def absmean(self, dim:Optional[int]=None, keepdim:bool=False):
        sum   = self.abssum(dim, keepdim)
        shape = self.shape

        if dim is None: return sum / torch.prod(shape)
        else: return sum / shape[dim]

    def mean(self, dim:Optional[int]=None, keepdim:bool=False):
        sum   = self.sum(dim, keepdim)
        shape = self.shape

        if dim is None: return sum / torch.prod(shape)
        else: return sum / shape[dim]

    def compress(self, n:int=1, dim:Optional[int]=None):
        if dim is None: dim = self.dim

        indices = []
        values  = []

        for shard, offset in zip(self.shards, self.limits[:-1]):
            mask = torch.argsort(shard.abs(), dim=dim, descending=True) < n
            values.append(shard[mask])

            i = torch.argwhere(mask)
            i[self.dim] += offset
            indices.append(i)

        indices = torch.concatenate(indices, dim=0).T
        values  = torch.concatenate(values, dim=0)
        print(indices.shape, values.shape)

        return torch.sparse_coo_tensor(indices, values, size=self.shape)

def checkattr(obj:object, name:str) -> bool:
    if hasattr(obj, name):
        return getattr(obj, name) is not None
    return False

def filter_noise(importance:torch.FloatTensor, window=slice(None), epsilon:float=1e-3, max_iter=20):
    '''A simple noise filter that enforces the following two constraints:
        1. The absolute sum of token saliencies within each class must be equal to one.
        2. The total saliency of a single token across all classes must sum to zero.

        As a result, if a token positively influences one class, it must negatively impact at least one other class.

        Args:
            importance:     Saliency map of the shape `num_classes` x `num_tokens`.
            window:         Optional window of tokens for the first requirement.
            epsilon:        Maximal error for each of the requirements (default: `1e-3`)

        Returns: Filtered saliencies
    '''
    for _ in range(max_iter):
        req1 = importance[:,window].sum(dim=0).abs() <= epsilon
        req2 = importance[:,window].abs().sum(dim=1) > (1.-epsilon)
        if (req1).all() and (req2).all(): break

        a  = importance
        a -= a.mean(dim=0)
        a /= a[:,window].abs().sum(dim=1, keepdim=True)

        b  = importance
        b /= b[:,window].abs().sum(dim=1, keepdim=True)
        b -= b.mean(dim=0)

        importance = .5 * (a + b)

    return importance

def embedding_backward(model:PreTrainedModel, dh:torch.Tensor, chunk_size:Optional[int]=None):
    # phi(input_ids, token_type_ids, token_ps) = W @ one_hot(input_ids) + f(token_type_ids, token_ps)
    #  => phi'(input_ids, token_type_ids, token_ps) = phi'(input_ids) = W
    #  
    # for f(input_ids, token_type_ids, token_ps) = nn(phi(input_ids, token_type_ids, token_ps)):
    #   => f'(input_ids, token_type_ids, token_ps) = nn'(input_ids, token_type_ids, token_ps) @ W

    dPhi = model.get_input_embeddings().weight.T.detach().to(dh)

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        m = (dh.shape[3] // n) + 1

        result = DistributedStackedTensor(
            shards=[torch.vmap(func=lambda grad: grad @ dPhi, chunk_size=chunk_size)(dh[:,:,:,i*m:(i+1)*m,:]) for i in range(n)],
            dim = 3
        )
        return result

    else: return torch.vmap(func=lambda grad: grad @ dPhi, chunk_size=chunk_size)(dh)

#=======================================================================#
# ContextManager for pausing explanations:                              #
#=======================================================================#

class no_explain:
    def __init__(self, target:Union['ChatGenerationPipeline', PreTrainedModel], reset_on_generate:bool=False, compute_grads:bool=False):
        assert hasattr(target, '_compute_grads') and hasattr(target, '_reset_on_generate')

        self.reset_on_generate = reset_on_generate
        self.compute_grads = compute_grads
        self.targets = [(target, target._compute_grads, target._reset_on_generate)]

        if isinstance(target, ChatGenerationPipeline):
            self.targets.append((target.model, target.model._compute_grads, target.model._reset_on_generate))

    def __enter__(self):
        for t, _, _ in  self.targets:
            t._compute_grads = self.compute_grads
            t._reset_on_generate = self.reset_on_generate

    def __exit__(self, *args):
        for t, g, r in  self.targets:
            t._compute_grads = g
            t._reset_on_generate = r

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

            # init basic properties:
            self._n               = 0           # initial prompt size
            self._x, self._y      = None, None  # last input + output
            self._a, self._da     = [], {}      # last attention maps + gradient
            self._h, self._dh     = [], {}      # last hidden states + gradient
            self._grad            = None        # gardient buffer

            self._grad_ids = {}

            self._reset_on_generate = True
            self._compute_grads = True

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
                dtype=torch.bfloat16, device=self._a[-1].device, requires_grad=False
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
                dtype=torch.bfloat16, device=self._h[-1].device, requires_grad=False
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

        def getOutputProbability(self, target:Iterable[str], precise:bool=True):
            '''Calculates the probability `p(target) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
            a specific output `target = [t_0, t_1, ..., t_n]` to happen.

            Args:
                target:     Iterable of tokens `t_i` for which to compute the probability.
                precise:    If `True`, calculate the probability `p(t_i|t_0...t_(i-1))` by running a sigle forward-pass for each `[t_0, ..., t_(i-1)]`. If `False`, assume `p(target) = p(t_0)` (default: `True`).

            Returns:        Token probabilities of the last next-token-prediction step with shape = (bs,)'''

            # get batch size:
            bs, _ = self._x.shape

            # p(target) = p(t_0)
            y = torch.nn.functional.softmax(self._y[:,0], dim=-1)
            p = y[:,self._grad_ids[target[0]]]
            x = torch.concatenate((self._x[:,:self._n], torch.full((bs, 1), self._grad_ids[target[0]], device=self._x.device, dtype=self._x.dtype)), dim=-1)

            if precise:
                # p(target) = p(t_0) * p(t_1|t_0) * ... * p(t_1|t_0...t_(j-1))
                with no_explain(self):
                    for t in target[1:]:
                        outputs = self.forward(x, use_cache=False)
                        y = torch.nn.functional.softmax(outputs.logits[:,-1,:].detach().clone().to(self._y.device, dtype=self._y.dtype), dim=-1)
                        p *= y[:,self._grad_ids[t]]
                        x = torch.concatenate((x, torch.full((bs, 1), self._grad_ids[t], device=self._x.device, dtype=self._x.dtype)), dim=-1)

            return p

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
                dtype=torch.bfloat16, device=self._a[-1].device, requires_grad=False
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
                dtype=torch.bfloat16, device=self._h[-1].device, requires_grad=False
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

        def grad(self, layer:int=0, outputs:Optional[Iterable[str]]=None, caching:bool=True):
            '''Jacobian with regard to the input of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
                caching:    If `True`, keep input gradient in memory for later use (default: `True`).

            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs, vocab_size)'''

            if (not caching) or (self._grad is None):
                grad = embedding_backward(self, self.getHiddenStateGradients(layer, outputs))
            else:
                grad = self._grad

            if caching and (self._grad is None):
                self._grad = grad
            
            return grad
        
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

        def iGrad(self, layer:int=0, alpha:float=1., beta:float=0., max_tokens:Optional[int]=None, outputs:Optional[Iterable[str]]=None, caching:bool=True):
            '''Inverted gradient scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                alpha:      Multiplicative offset for the decision boundary (default 1.).
                beta:       Additive offset for the decision boundary (default 0.).
                max_tokens: If specified, only evaluates for the `max_tokens` input tokens with the highest variation.
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).
                caching:    If `True`, keep input gradient in memory for later use (default: `True`).
                
            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs, vocab_size)'''

            if outputs is None: outputs = self._grad_ids

            # compute jacobian:
            dx = self.grad(layer, outputs=outputs, caching=caching)

            # extract dimensions (bs x n_classes x n_outputs x n_inputs x vocab_size):
            bs, n_classes, n_outputs, n_inputs, vocab_size = dx.shape

            # reduce size:
            if max_tokens is not None:
                if vocab_size > max_tokens:
                    vocab_size = max_tokens
                    token_ids  = self._x.unique()
                    token_ids  = torch.concat(
                        (token_ids, torch.argsort((dx.max(3) - dx.min(3)).mean(dim=(0,1,2,3)), descending=True)[:vocab_size-token_ids.shape[0]]),
                        dim = 0
                    )
                    dx = dx[:,:,:,:,token_ids]

            # reshape to (n_classes x (bs x n_outputs x n_inputs x vocab_size)):
            dx = dx.transpose(0,1).flatten(start_dim=1)

            # invert jacobian (shape: bs x n_outputs x n_inputs x vocab_size x n_classes)):
            dx = -torch.linalg.pinv(dx.to(float)).view((bs, n_outputs, n_inputs, vocab_size, n_classes))

            # get last output (shape: bs x n_outputs x n_classes):
            y = torch.stack([self._y[:,:,self._grad_ids[key]] for key in outputs], dim=-1)

            # calculate distance on outputs (shape: bs x n_outputs x n_classes):
            diff_y = y.max(dim=-1, keepdim=True).values - y

            # scale and shift:
            diff_y *= alpha
            diff_y += beta

            # calculate distance on input (shape: bs x n_classes x n_outputs x n_inputs x vocab_size):
            diff_h = torch.zeros((bs, n_classes, n_outputs, n_inputs, vocab_size), device=dx.device, dtype=dx.dtype)
            for i in range(bs):
                for j in range(n_outputs):
                    for c in range(n_classes):
                        diff_h[i, c, j, :, :] = dx[i, j, :, :, c] * diff_y[i, j, c]

            if max_tokens is None: return diff_h
            else: return diff_h, token_ids

        def gradH(self, layer:int=0, outputs:Optional[Iterable[str]]=None):
            '''GradIn (`dh ⊙ h`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).

            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs, encoding_size)'''

            # compute hidden states and gradients:
            h  = self.hiddenStates[:,layer]
            dh = self.getHiddenStateGradients(layer, outputs)

            return dh * torch.unsqueeze(h, 1)

        def gradIn(self, layer:int=0, outputs:Optional[Iterable[str]]=None, caching:bool=True):
            '''GradIn (`dx · x`) scores of the last batch.

            Args:
                layer:      Transformer layer to compute the scores for (default 0).
                outputs:    Iterable of output names to compute the scores for. Names can be registered by calling `registerGradientComputation(...)` (optional).
                caching:    If `True`, keep input gradient in memory for later use (default: `True`).
 
            Returns:        Importance scores with shape = (bs, n_classes, n_outputs, n_inputs)'''

            # compute hidden states and gradients:
            dx = self.grad(layer, outputs=outputs, caching=caching)

            # multiplication with a one-hot tensor can be replaced with indexing:
            return torch.stack([
                dx[i, :, :, torch.arange(len(tokens)), tokens]
                for i, tokens in enumerate(self._x)
            ])

        def forward(self, input_ids:torch.Tensor, *args, **kwargs):
            # reset gradient buffer:
            if self._reset_on_generate: self._grad = None

            if self._compute_grads: 
                # control gradient computation:
                prev_grad = torch.is_grad_enabled()
                if len(self._grad_ids) > 0: torch.set_grad_enabled(True)
                
                # propagate through model:
                outputs = super().forward(input_ids, *args, **kwargs)

                has_attentions    = checkattr(outputs, 'attentions')
                has_hidden_states = checkattr(outputs, 'hidden_states')

                # save gradients:
                if len(self._grad_ids) > 0:
                    output_size = outputs.logits.shape[-1]

                    for key in self._grad_ids:
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

                        y = (outputs.logits[:,-1] @ one_hot(self._grad_ids[key], output_size).to(outputs.logits)).sum()

                        # calculate gradients of output:
                        y.backward(retain_graph = True)

                        # save gradients with regard to attention weights:
                        # Iterable of n_output tensors of shape (bs x n_layers x n_heads x n_inputs)
                        if has_attentions:
                            if key not in self._da: self._da[key] = []
                            self._da[key].append(torch.stack(
                                # outputs.attentions is a list of n_layers tensors of shape (bs x n_heads x sequence_length, sequence_length)
                                [a.grad[:,:,-1,:].detach().clone().to(device=self._device, dtype=torch.bfloat16) for a in outputs.attentions],
                                dim=1 
                            ))

                        # save gradients with regard to hidden states:
                        # Iterable of n_output tensors of shape (bs x n_layers x n_inputs x encoding_size)
                        if has_hidden_states:
                            if key not in self._dh: self._dh[key] = []
                            self._dh[key].append(torch.stack(
                                [h.grad.detach().clone().to(device=self._device, dtype=torch.bfloat16) for h in outputs.hidden_states],
                                dim=1 
                            ))

                # save attentions:
                # Iterable of n_output tensors of shape (bs x n_layers x n_heads x n_inputs)
                if has_attentions:
                    self._a.append(torch.stack(
                        # outputs.attentions is a list of n_layers tensors of shape (bs x n_heads x sequence_length, sequence_length)
                        [a[:,:,-1,:].detach().clone().to(device=self._device, dtype=torch.bfloat16) for a in outputs.attentions],
                        dim=1 
                    ))

                # save hidden states:
                # Iterable of n_output tensors of shape (bs x n_layers x n_inputs x encoding_size)
                if has_hidden_states:
                    self._h.append(torch.stack(
                        [h.detach().clone().to(device=self._device, dtype=torch.bfloat16) for h in outputs.hidden_states],
                        dim=1 
                    ))

                # reset gradient computation:
                if len(self._grad_ids) > 0: torch.set_grad_enabled(prev_grad)
            
            # skip gradient computation if compute_grads flag is not set:
            else: outputs = super().forward(input_ids, *args, **kwargs)

            # save last sequence:
            if self._reset_on_generate:
                if self._x is None: self._x = input_ids.detach().clone().to(device=self._device)
                else: self._x = torch.concat([self._x, input_ids.detach().clone().to(device=self._device)], dim=1)

                if self._y is None: self._y = outputs.logits[:,-1:,:].detach().clone().to(device=self._device, dtype=torch.bfloat16)
                else: self._y = torch.concat([self._y, outputs.logits[:,-1:,:].detach().clone().to(device=self._device, dtype=torch.bfloat16)], dim=1)

            return outputs

        def generate(self, input_ids:torch.Tensor, outputs:Dict[str, int]={}, **kwargs):
            ''' Generates sequences of token ids for models with a language modeling head and activates gradient computation for specific outputs so that their corresponding explanations can be computed.

            Args:
                input_ids:      The sequence used as a prompt for the generation or as model inputs to the encoder.
                outputs:        Dictionary of output names and indices to compute the scores for (optional).
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
            # skip gradient computation if no_explain flag is set:
            if self._reset_on_generate:

                # get least used gpu if cuda is available:
                if not hasattr(self, '_device'):
                    if torch.cuda.is_available():
                        space = torch.tensor([torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())], device='cpu')
                        self._device = torch.device(f'cuda:{torch.argmax(space):d}')

                    else: self._device = torch.device('cpu')

                # register output ids for gradient computation:
                self._grad_ids = {
                    key:torch.tensor(outputs[key], dtype=int)
                    for key in outputs
                }

                # reset state:
                self._n               = input_ids.shape[1]  # initial input size
                self._x, self._y      = None, None          # last input + output
                self._a, self._da     = [], {}              # last attention maps + gradient
                self._h, self._dh     = [], {}              # last hidden states + gradient

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

        # get prefix and suffix sizes:
        ids = self.tokenizer(self.getPrompt(self.mask_token), add_special_tokens=False, return_attention_mask=False)['input_ids']
        self.prompt_prefix_size = ids.index(self.mask_token_id)
        self.prompt_suffix_size = ids[::-1].index(self.mask_token_id)

        # cached variables:
        self._reset_on_generate = True
        self._compute_grads = True
        self._token_cache = []
        self._last_input = ['']

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

    def _aggregate_saliency(self, inputs:torch.Tensor, token_indices:List[List[int]], precise:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''Aggregates saliency maps per label.

        Args:
            inputs:         Input `torch.Tensor` of shape (n_tokens, n_outputs, n_inputs, ...).
            token_indices:  Token indices per label as provided in the second output of `_outputs2set(...)`.
            precise:        If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            aggregation:    Defines how token saliencies are aggregated: `'saliency'` computes the weighted mean by summed saliencies towards an output token, `'cumsum'` computes the cumsum and normailzes by number of output tokens, and `'equal'`computes the unqweighted mean.

        Returns:            Aggregated saliency scores with shape = (n_classes, n_inputs, ...)'''
        # precise == True -> aggregate per label:
        if precise:
            n_outputs = inputs.shape[1]
            inputs    = [torch.stack([
                            inputs[i, j, :] for j, i in enumerate(indices[:n_outputs])
                        ], dim=0) for indices in token_indices]

            # aggregate scores per label (shape of s: n_tokens, n_inputs, ...):
            for i, s in enumerate(inputs):
                m = s.shape[0]
                n = s.shape[1] - n_outputs

                inputs[i] = s[:,:n]

                if aggregation == 'saliency':
                    inputs[i][:-1] *= (1. + s[:,torch.arange(m-1)+n+1].sum(dim=0, keepdim=True).transpose(0, 1))

                elif aggregation == 'cumsum':
                    inputs[i] = torch.cumsum(inputs[i], dim=0)

                elif aggregation == 'equal': pass
                else: raise ValueError(f"Parameter `aggregation` must be either 'saliency', 'cumsum', or 'equal' but is '{str(aggregation)}'.")

                inputs[i] = inputs[i].mean(dim=0)

            return torch.stack(inputs, dim=0)

        # precise == False -> only use first predicted token:
        else: return torch.stack(
            [inputs[indices[0], 0, :] for indices in token_indices],
            dim=0
        )

    @property
    def mask_token(self) -> str:
        return self.tokenizer.unk_token

    @property
    def mask_token_id(self) -> int:
        return self.tokenizer.convert_tokens_to_ids(self.mask_token)

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, max_seq_len:int, max_gen_len:int, **kwargs):
        prefix = pretrained_model_name_or_path.split('/')[0].lower()
        if prefix == "meta-llama":  return LlamaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        if prefix == "google":      return GemmaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        if prefix == "mistralai":   return MistralChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        if prefix == "deepseek-ai": return LlamaChatGenerationPipeline.from_pretrained(pretrained_model_name_or_path, max_seq_len, max_gen_len, **kwargs)
        raise NotImplementedError()

    def getOutputProbability(self, target:Union[Iterable[str], str], precise:bool=True):
        '''Calculates the probability `p(target) = p(t_0) * p(t_1|t_0) * ... * p(t_n|t_0...t_(n-1))` of
        a specific output `target = [t_0, t_1, ..., t_n]` to happen.

        Args:
            target:     String or iterable of tokens `t_i` for which to compute the probability.
            precise:    If `True`, calculate the probability `p(t_i|t_0...t_(i-1))` by running a sigle forward-pass for each `[t_0, ..., t_(i-1)]`. If `False`, assume `p(target) = p(t_0)` (default: `True`).

        Returns:        Probability of the target to be predicted during the last call to `.generate(...)`'''
        # convert string to Iterable of tokens:
        if isinstance(target, str):
            target = self.tokenizer.convert_ids_to_tokens(self.tokenizer(target)['input_ids'][1:])

        # call base method:
        return self.model.getOutputProbability(target=list(target), precise=precise)[0]

    def grad(self, layer:int=0, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, precise:bool=True, caching:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''Jacobian with regard to the input of the last batch.

        Args:
            layer:          Transformer layer to compute the scores for (default 0).
            outputs:        Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            precise:        If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            caching:        If `True`, keep input gradient in memory for later use (default: `True`).
            aggregation:    Defines how token saliencies are aggregated: `'saliency'` computes the weighted mean by summed saliencies towards an output token, `'cumsum'` computes the cumsum and normailzes by number of output tokens, and `'equal'`computes the unqweighted mean.

        Returns:        Importance scores with shape = (n_classes, n_inputs, vocab_size)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores (shape: n_tokens, n_outputs, n_inputs, vocab_size):
        grad = self.model.grad(layer=layer, outputs=tokens, caching=caching)[0]

        # aggregate saliency score per label (shape: n_classes, n_inputs, vocab_size):
        return self._aggregate_saliency(inputs=grad, token_indices=token_indices, precise=precise, aggregation=aggregation)

    def aGrad(self, layer:int=-1, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, precise:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''AGrad (`-da ⊙ a`) scores of the last batch.

        Args:
            layer:          Transformer layer to compute the scores for (default -1).
            outputs:        Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            precise:        If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            aggregation:    Defines how token saliencies are aggregated: `'saliency'` computes the weighted mean by summed saliencies towards an output token, `'cumsum'` computes the cumsum and normailzes by number of output tokens, and `'equal'`computes the unqweighted mean.

        Returns:        Importance scores with shape = (n_heads, n_classes, n_inputs)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores (shape: n_heads, n_tokens, n_outputs, n_inputs):
        aGrad = self.model.aGrad(layer=layer, outputs=tokens)[0]

        # change shape to (n_tokens, n_outputs, n_inputs, n_heads):
        aGrad = torch.permute(aGrad, (1,2,3,0))

        # aggregate saliency score per label (shape: n_classes, n_inputs, n_heads):
        aGrad = self._aggregate_saliency(inputs=aGrad, token_indices=token_indices, precise=precise, aggregation=aggregation)

        # change shape to (n_heads, n_classes, n_inputs):
        return torch.permute(aGrad, (2,0,1))

    def repAGrad(self, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, precise:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''RepAGrad scores of the last batch.

        Args:
            outputs:        Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            precise:        If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            aggregation:    Defines how token saliencies are aggregated: `'saliency'` computes the weighted mean by summed saliencies towards an output token, `'cumsum'` computes the cumsum and normailzes by number of output tokens, and `'equal'`computes the unqweighted mean.

        Returns:        Importance scores with shape = (n_heads, n_classes, n_inputs)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores (shape: n_heads, n_tokens, n_outputs, n_inputs):
        repAGrad = self.model.repAGrad(outputs=tokens)[0]

        # change shape to (n_tokens, n_outputs, n_inputs, n_heads):
        repAGrad = torch.permute(repAGrad, (1,2,3,0))

        # aggregate saliency score per label (shape: n_classes, n_inputs, n_heads):
        repAGrad = self._aggregate_saliency(inputs=repAGrad, token_indices=token_indices, precise=precise, aggregation=aggregation)

        # change shape to (n_heads, n_classes, n_inputs):
        return torch.permute(repAGrad, (2,0,1))

    def iGrad(self, layer:int=0, alpha:float=1., beta:float=0., max_tokens:Optional[int]=None, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, precise:bool=True, caching:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''Inverted gradient scores of the last batch.

        Args:
            layer:          Transformer layer to compute the scores for (default 0).
            alpha:          Multiplicative offset for the decision boundary.
            beta:           Additive offset for the decision boundary.
            max_tokens:     If specified, only evaluates for the `max_tokens` input tokens with the highest variation.
            outputs:        Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            precise:        If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            caching:        If `True`, keep input gradient in memory for later use (default: `True`).
            aggregation:    Defines how token saliencies are aggregated: `'saliency'` computes the weighted mean by summed saliencies towards an output token, `'cumsum'` computes the cumsum and normailzes by number of output tokens, and `'equal'`computes the unqweighted mean.
 
        Returns:        Importance scores with shape = (n_classes, n_inputs, vocab_size)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores (shape: n_tokens, n_outputs, n_inputs, encoding_size):
        iGrad = self.model.iGrad(layer=layer, alpha=alpha, beta=beta, max_tokens=max_tokens, outputs=tokens, caching=caching)

        # aggregate saliency score per label (shape: n_classes, n_inputs, encoding_size):
        if max_tokens is None: return self._aggregate_saliency(inputs=iGrad[0], token_indices=token_indices, precise=precise, aggregation=aggregation)
        else: return self._aggregate_saliency(inputs=iGrad[0][0], token_indices=token_indices, precise=precise, aggregation=aggregation), self.tokenizer.convert_ids_to_tokens(iGrad[1])

    def gradIn(self, layer:int=0, window:slice=slice(None), epsilon:float=1e-3, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, precise:bool=True, normalize:bool=True, caching:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''GradIn (`dx · x`) scores of the last batch.

        Args:
            layer:      Transformer layer to compute the scores for (default 0).
            window:     Optional window of tokens for which the saliencies need to sum to 1. after noise filtering.
            epsilon:    Maximal error for each of the requirements of noise filtering (default: `1e-3`)
            outputs:    Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            precise:    If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            normalize:  Use noise filtering (default: `True`).
            caching:    If `True`, keep input gradient in memory for later use (default: `True`).
            
        Returns:        Importance scores with shape = (n_classes, n_inputs)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores (shape: n_tokens, n_outputs, n_inputs):
        gradIn = self.model.gradIn(layer=layer, outputs=tokens, caching=caching)[0]

        # aggregate saliency score per label (shape: n_classes, n_inputs):
        gradIn = self._aggregate_saliency(inputs=gradIn, token_indices=token_indices, precise=precise, aggregation=aggregation)

        # filter noise:
        return filter_noise(gradIn, window=window, epsilon=epsilon) if normalize else gradIn
    
    def gradH(self, layer:int=0, window:slice=slice(None), epsilon:float=1e-3, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, precise:bool=True, normalize:bool=True, aggregation:Union[Literal['saliency'], Literal['cumsum'], Literal['equal']]='saliency'):
        '''GradH (`dh ⊙ h`) scores of the last batch.

        Args:
            layer:          Transformer layer to compute the scores for (default 0).
            window:         Optional window of tokens for which the saliencies need to sum to 1. after noise filtering.
            epsilon:        Maximal error for each of the requirements of noise filtering (default: `1e-3`)
            outputs:        Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            precise:        If `True`, labels are aggregated. If `False` returns the saliency towards the first output token (default: `True`).
            normalize:      Use noise filtering (default: `True`).
            aggregation:    Defines how token saliencies are aggregated: `'saliency'` computes the weighted mean by summed saliencies towards an output token, `'cumsum'` computes the cumsum and normailzes by number of output tokens, and `'equal'`computes the unqweighted mean.

        Returns:        Importance scores with shape = (n_classes, n_inputs)'''

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # compute scores (shape: n_tokens, n_outputs, n_inputs):
        gradH = self.model.gradH(layer=layer, outputs=tokens)[0].sum(dim=-1)

        # aggregate saliency score per label (shape: n_classes, n_inputs):
        gradH = self._aggregate_saliency(inputs=gradH, token_indices=token_indices, precise=precise, aggregation=aggregation)

        # filter noise:
        return filter_noise(gradH, window=window, epsilon=epsilon) if normalize else gradH

    def shap(self, outputs:Union[Iterable[str],Iterable[Iterable[str]],None]=None, max_new_tokens:Optional[int]=None, fixed_prefix:str='', fixed_suffix:str='', return_explanation:bool=False, **kwargs):
        '''Shap scores of the last batch.

        Args:
            outputs:            Iterable of output strings to compute the scores for. Strings can be registered by calling `generate(...)` with the parameter `compute_gradients` set (optional).
            fixed_prefix:       Prefix that will not be tested by the explainer.
            fixed_suffix:       Postfix that will not be tested by the explainer.
            max_new_tokens:     The maximum numbers of tokens to generate, ignore the current number of tokens.
            return_explanation: Return a `shap.Explanation` object instead of a sailency map.
            
        Returns:                Importance scores with shape = (n_classes, n_inputs)'''

        # make sure the last input starts and ends with the specified pre- and suffixes:
        for s in self._last_input:
            assert isinstance(fixed_prefix, str) and s.startswith(fixed_prefix), s
            assert isinstance(fixed_suffix, str) and s.endswith(fixed_suffix), s

        # analyze input:
        tokens, token_indices = self._outputs2set(outputs)

        # build list of labels (including possible label parts):
        labels = []
        for indices in token_indices:
            # build list of possible snippets:
            snippets = [[]]
            for i in indices: snippets.append(
                snippets[-1] + [self.tokenizer.vocab[tokens[i]]]
            )

            # detokenize and append to labels:
            labels.append([self.tokenizer.decode(ids).lower() for ids in snippets[1:]])

        # create model function:
        def f(s_in:List[str]):
            probabilities = np.zeros((len(s_in), len(labels)), dtype=float)

            # generate outputs:
            s_out = [chat[-1][-1] for chat in self.generate(s_in,
                output_attentions=False,
                output_hidden_states=False,
                max_new_tokens=max_new_tokens
            )[0]]

            for i, s in enumerate(s_out):
                s = s.strip(' "\'*').lower()

                for j, label in enumerate(labels):
                    score = 0.
                    for snippet in label:
                        score += float(s.startswith(snippet))

                    probabilities[i,j] = score / len(label)

            return probabilities

        with no_explain(self):
            # run shap:
            masker = shap.maskers.Text(self.tokenizer, mask_token=self.mask_token, collapse_mask_token=True)
            masker.keep_prefix = len(self.tokenizer(fixed_prefix, add_special_tokens=False, return_attention_mask=False)['input_ids'])
            masker.keep_suffix = len(self.tokenizer(fixed_suffix, add_special_tokens=False, return_attention_mask=False)['input_ids'])

            explainer   = shap.PartitionExplainer(f,
                            masker=masker,
                            output_names=[l[-1] for l in labels],
                            **kwargs
                        )

            explanation = explainer(self._last_input)

        # return explanations:
        if return_explanation: return explanation
        else: return torch.concatenate((
            torch.zeros((len(labels), self.prompt_prefix_size), dtype=float),
            torch.tensor(explanation.values[0].T / (np.abs(explanation.values).max() + 1e-9)),
            torch.zeros((len(labels), self.prompt_suffix_size), dtype=float)
        ), dim=1)

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

    def generate(self, txt:Union[str, List[str]], history:Optional[torch.Tensor]=None, compute_grads:Union[Iterable[str],Iterable[Iterable[str]]]=[], **kwargs):
        # convert input to numpy:
        single_input = isinstance(txt, str)
        if single_input: txt = [txt]
        txt = np.array(txt, dtype=object)
        
        # save input for shap:
        if self._reset_on_generate: self._last_input = txt

        # format prompt:
        if history is None: prompt = np.vectorize(self.getPrompt)(txt)
        else: prompt = np.vectorize(lambda s: self._eot + self.getPrompt(s))(txt)

        # tokenize:
        input_ids = [
            torch.tensor(ids, dtype=int, device=self.model.device)
            for ids in self.tokenizer(prompt.tolist(), return_attention_mask=False)["input_ids"]
        ]

        # append history and pad:
        bs, seq_len = len(input_ids), max([len(ids) for ids in input_ids])
        if history is not None: seq_len += history.shape[1] - 2

        input_ids_pad  = torch.full((bs, seq_len), self.tokenizer.pad_token_id, dtype=int, device=self.model.device)
        attention_mask = torch.zeros((bs, seq_len), dtype=int, device=self.model.device)
        for i, ids in enumerate(input_ids):
            if history is not None:
                ids = torch.concatenate((history[i,:-1], ids[1:]))

            n = ids.shape[0]
            input_ids_pad[i, -n:] = ids
            attention_mask[i, -n:] = 1

        # generate and register gradient computation:
        kwargs.update(self.model_kwargs)
        if self._reset_on_generate:
            self._token_cache = self._outputs2set(compute_grads)
            outputs = {token: self.tokenizer.vocab[token] for token in self._token_cache[0]}
        else: outputs = {}
        output_ids = self.model.generate(
            input_ids=input_ids_pad,
            attention_mask=attention_mask,
            outputs=outputs,
            **kwargs
        )

        # decode chats:
        chats = [self._rex.findall(self.tokenizer.decode(ids[:-1]) + self._eot) for ids in output_ids]

        #return:
        if single_input: return chats[0], input_ids_pad, output_ids
        else: return chats, input_ids_pad, output_ids

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
        model_type = None
        if pretrained_model_name_or_path.startswith("google/gemma-1"): model_type = GemmaForCausalLM
        if pretrained_model_name_or_path.startswith("google/gemma-2"): model_type = Gemma2ForCausalLM
        
        return GemmaChatGenerationPipeline(
            tokenizer   = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model       = GenericModelForExplainableCausalLM(model_type).from_pretrained(pretrained_model_name_or_path, **kwargs),
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
        if pad_token_id is None: pad_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')

        super().__init__(
            tokenizer    = tokenizer,
            model        = model,
            bos          = '<|begin_of_text|>',
            sot          = lambda role: f'<|start_header_id|>{role}<|end_header_id|>\n\n',
            eot          = '<|eot_id|>',
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            pad_token_id = pad_token_id
        )

        self.model.generation_config = GenerationConfig(
            bos_token_id = tokenizer.convert_tokens_to_ids('<|begin_of_text|>'),
            eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            pad_token_id = tokenizer.convert_tokens_to_ids('<|eot_id|>') if pad_token_id is None else pad_token_id,
            
            max_length = max_seq_len,
            max_new_tokens = max_gen_len,
        )

        self.tokenizer.pad_token_id = pad_token_id

    @property
    def mask_token(self):
        return '###'
        #return '<|reserved_special_token_250|>'

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
    def mask_token(self):
        return '<unk>'

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path:str, **kwargs):
        return LlamaChatGenerationPipeline(
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path),
            model     = GenericModelForExplainableCausalLM(LlamaForCausalLM).from_pretrained(pretrained_model_name_or_path, **kwargs)
        )