import collections
import numpy as np
import torch
import pdb, einops

DTYPE = torch.float
DEVICE = 'cuda:0'

#-----------------------------------------------------------------------------#
#------------------------------ numpy <--> torch -----------------------------#
#-----------------------------------------------------------------------------#

def to_np(x):
	if torch.is_tensor(x):
		x = x.detach().cpu().numpy()
	return x

def to_torch(x, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x.items()}
	elif torch.is_tensor(x):
		return x.to(device).type(dtype)
		# import pdb; pdb.set_trace()
	return torch.tensor(x, dtype=dtype, device=device)

def to_device(x, device=DEVICE):
	if torch.is_tensor(x):
		return x.to(device)
	elif type(x) is dict:
		return {k: to_device(v, device) for k, v in x.items()}
	else:
		print(f'Unrecognized type in `to_device`: {type(x)}')
		pdb.set_trace()
	# return [x.to(device) for x in xs]

def to_device_tp(*args, device):
    ## args is a tuple
    return tuple(to_device(arg, device=device) for arg in args)
	

# def atleast_2d(x, axis=0):
# 	'''
# 		works for both np arrays and torch tensors
# 	'''
# 	while len(x.shape) < 2:
# 		shape = (1, *x.shape) if axis == 0 else (*x.shape, 1)
# 		x = x.reshape(*shape)
# 	return x

# def to_2d(x):
# 	dim = x.shape[-1]
# 	return x.reshape(-1, dim)

def batchify(batch):
	'''
		convert a single dataset item to a batch suitable for passing to a model by
			1) converting np arrays to torch tensors and
			2) and ensuring that everything has a batch dimension
	'''
	fn = lambda x: to_torch(x[None])

	batched_vals = []
	for field in batch._fields:
		val = getattr(batch, field)
		val = apply_dict(fn, val) if type(val) is dict else fn(val)
		batched_vals.append(val)
	return type(batch)(*batched_vals)

def batch_copy(batch, n_copy):
	"""
	copy batch size 1 to n_copy, just to fill the shape for debugging
	dim should be (B, ...)
	can implement a concat two batches version
	"""
	def fn(x):
		assert x.shape[0] == 1
		return torch.repeat_interleave(x, n_copy, dim=0)

	batched_vals = []
	for field in batch._fields:
		val = getattr(batch, field)
		val = apply_dict(fn, val) if type(val) is dict else fn(val)
		batched_vals.append(val)
	
	return type(batch)(*batched_vals)


def batchify_seq(batches: list):
	'''
		support batch size > 1
		return a tuple
		not finished yet
	'''
	assert False
	n_elem = len(batches[0])
	bat_seq = [] # a list of tuple
	for i_b, batch in enumerate(batches):
		bat_seq.append(batchify(batch))
	
	pdb.set_trace()
	outs = []
	for i_e in range(n_elem):
		out = []
		## loop through all batches
		for i_x, bat in enumerate(bat_seq):
			out.append( bat[i_e] )
		
		## BUG: not finished
		assert False
		if type(bat[i_e]) == dict:
			out_d = {}
			for kk in bat[i_e]:
				for i_x, bat in enumerate(bat_seq):
					# out.append( bat[i_e][] )
					pass

		# else:
		out = torch.cat(out, dim=0)
		outs.append(out)
	
	return tuple(outs)



def apply_dict(fn, d, *args, **kwargs):
	return {
		k: fn(v, *args, **kwargs)
		for k, v in d.items()
	}

def normalize(x):
	"""
		scales `x` to [0, 1]
	"""
	x = x - x.min()
	x = x / x.max()
	return x

def to_img(x):
    normalized = normalize(x)
    array = to_np(normalized)
    array = np.transpose(array, (1,2,0))
    return (array * 255).astype(np.uint8)

def set_device(device):
	DEVICE = device
	if 'cuda' in device:
		torch.set_default_tensor_type(torch.cuda.FloatTensor)

def batch_to_device(batch, device='cuda:0'):
    vals = [
        to_device(getattr(batch, field), device)
        for field in batch._fields
    ]
    return type(batch)(*vals)

def _to_str(num):
	if num >= 1e6:
		return f'{(num/1e6):.2f} M'
	else:
		return f'{(num/1e3):.2f} k'

#-----------------------------------------------------------------------------#
#----------------------------- parameter counting ----------------------------#
#-----------------------------------------------------------------------------#

def param_to_module(param):
	module_name = param[::-1].split('.', maxsplit=1)[-1][::-1]
	return module_name

def report_parameters(model, topk=10):
	counts = {k: p.numel() for k, p in model.named_parameters()}
	n_parameters = sum(counts.values())
	print(f'[ utils/arrays ] Total parameters: {_to_str(n_parameters)}')

	modules = dict(model.named_modules())
	sorted_keys = sorted(counts, key=lambda x: -counts[x])
	max_length = max([len(k) for k in sorted_keys])
	for i in range(topk):
		key = sorted_keys[i]
		count = counts[key]
		module = param_to_module(key)
		print(' '*8, f'{key:10}: {_to_str(count)} | {modules[module]}')

	remaining_parameters = sum([counts[k] for k in sorted_keys[topk:]])
	print(' '*8, f'... and {len(counts)-topk} others accounting for {_to_str(remaining_parameters)} parameters')
	return n_parameters


def batch_repeat_tensor_in_dict(x: torch.Tensor, t_2d: torch.Tensor, cond_dd: dict, n_rp: int):
	'''
	FIXED: Return a new dict rather than modified the original Dict!
	'''
	if x is not None:
		## B H D
		x = x.repeat( (n_rp, 1, 1) )
	if t_2d is not None:
		assert t_2d.ndim == 2
		t_2d = t_2d.repeat( (n_rp, 1,) )
	
	new_dd = {}
	for k in cond_dd.keys():
		if torch.is_tensor(cond_dd[k]):
			new_dd[k] = cond_dd[k].repeat(   [n_rp] + [1,] * len(cond_dd[k].shape[1:])  )
		elif type(cond_dd[k]) == np.ndarray:
			# dd[k]
			new_dd[k] = einops.repeat(cond_dd[k], 'b ... -> (rr b) ...', rr=n_rp )
		else:
			new_dd[k] = cond_dd[k]
			assert type(cond_dd[k]) in [bool, type(None)]

			
	return x, t_2d, new_dd
    # x = x.repeat( (n_rp, 1, 1) )
    # cond_2 = {}
    # for k in cond:
    #     cond_2[k] = cond[k].repeat( (n_rp, 1,) ) # 2d (B,2)
    # t = t.repeat( (n_rp,) ) # (B,)
    # w = w.repeat( (n_rp, 1,) ) # 2d

    # return x, cond_2, t, w
    