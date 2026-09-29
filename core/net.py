from dataclasses import dataclass
import jax, flax, orbax
import jax.numpy as jnp
from flax import linen as nn
import orbax.checkpoint as ocp
from etils import epath


class MyMLP(nn.Module):
    hidden_width: int
    output_features: int    
    hidden_layers: int
    add_bn: bool = True
    sigmoid_shrink: float = 3     
    hidden_act_type: str = 'sigmoid'  # relu
    do_dropout: bool = False
    dropout_rate: float = 0.4
    net_init_seed: int = 0

    def setup(self):
        # Note: it's only semantically a "conv" - 1,1,1 kernel means it's just a voxelwise MLP
        self.conv1 = nn.Conv(
            features=self.hidden_width, kernel_size=(1, 1, 1), padding="SAME", kernel_init=nn.initializers.xavier_uniform()
        )
        self.bln1 = nn.BatchNorm()
        
        self.internal_convs = [
            nn.Conv(
                features=self.hidden_width, 
                kernel_size=(1, 1, 1), 
                padding="SAME", 
                kernel_init=nn.initializers.xavier_uniform()
            ) for jj in range(self.hidden_layers)
        ]
        self.internal_BNs = [nn.BatchNorm() for jj in range(self.hidden_layers)]
        
        self.conv_out = nn.Conv(
            features=self.output_features, kernel_size=(1, 1, 1), padding="SAME", kernel_init=nn.initializers.xavier_uniform()
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
        if self.hidden_act_type == 'sigmoid':
            self.inner_activation = nn.sigmoid
        elif self.hidden_act_type == 'relu':
            self.inner_activation = jax.nn.relu
        else:
            raise NotImplemented('unknown activation')
        
    def __call__(self, inputs, train=True, enable_dropout=None):
        x = self.conv1(inputs)
        if self.add_bn:
            x = self.bln1(x, use_running_average=not train)
        x = self.inner_activation(x)        
        if enable_dropout is None:
            enable_dropout = train
        if self.do_dropout:
            x = self.dropout(x, deterministic=not enable_dropout)
            
        for hli in range(self.hidden_layers):
            x = self.internal_convs[hli](x)
            if self.add_bn:
                x = self.internal_BNs[hli](x, use_running_average=not train)
            x = self.inner_activation(x)        
            if self.do_dropout:
                x = self.dropout(x, deterministic=not enable_dropout)
        
        x = self.conv_out(x) 
        # conservative bias towards mid-range by a given factor                
        x = nn.sigmoid(x/self.sigmoid_shrink)
            
        return x
        

def get_net(input_shape, mrf_len=30, extra_inputs=3, **kwargs):     
    """
        input features: 33 [35] = mrf_len (30) + extra_inputs ( T1 + T2 + B1 + [fss, kss] (for cest given MT))                      
        output features: 6 = fs[s], ks[s], and the optional/experimental b1fix, r2cfix, R2fix
    """
    model = MyMLP(**kwargs) 
    
    # -- Initialize the model with dummy input --
    # jax convention: input_shape = (batch_size, height, width, depth, inp_channels)
    input_shape = [1] + list(input_shape) + [mrf_len + extra_inputs]  
    dummy_input = jnp.ones(input_shape)
    params = model.init(jax.random.PRNGKey(model.net_init_seed), dummy_input, train=True)

    return model, params


@dataclass
class ModelState:
    apply_fn:callable = None
    params:dict = None
    batch_stats:dict = None
    
    
def state2predictor(model_state, frozen_BN_infer=False, **kwargs):    
    params = {'params': model_state.params, 'batch_stats': model_state.batch_stats}
    if frozen_BN_infer:
        nn_predictor = lambda x: model_state.apply_fn(params, x, train=False, mutable=[], **kwargs)
    else:
        nn_predictor = lambda x: model_state.apply_fn(params, x, train=True, mutable=['batch_stats'], **kwargs)
    return nn_predictor


def get_ckpt_mngr(folder):
    options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        save_interval_steps=2,
        cleanup_tmp_directories=True, # overwrite?
        create=True  
    )
    mngr = ocp.CheckpointManager(
        epath.Path(folder),
        {
            'model_state': ocp.PyTreeCheckpointer(),
            'config': ocp.PyTreeCheckpointer()
        },
    options=options)
    
    return mngr

def save_ckpt(model_state, config={'train_cfg':{}, 'net_cfg':{}}, folder='/tmp/mymodel/', step=666):
    
    mngr = get_ckpt_mngr(folder)
    mngr.save(step, {'model_state': model_state, 'config': config})    
    

def load_ckpt(folder='/tmp/mymodel/', **kwargs): #, **get_net_params):

    mngr = get_ckpt_mngr(folder)
    step = mngr.latest_step()  
    restored = mngr.restore(step)

    model_state_d = restored['model_state']
    config = restored['config']

    get_net_kwargs = config['net_cfg']
    
    nnmodel, nnparams = get_net(**get_net_kwargs)
    
    _nn_predictor = state2predictor(ModelState(
        apply_fn=nnmodel.apply, 
        params=model_state_d['params'],
        batch_stats=model_state_d['batch_stats']
        ), **kwargs)    
    
    return _nn_predictor, config


def get_model_state(folder='/tmp/mymodel/'):
    mngr = get_ckpt_mngr(folder)    
    restored = mngr.restore(mngr.latest_step())    
    model_state_d = restored['model_state']
    nnmodel, _ = get_net(**restored['config']['net_cfg'])
    return ModelState(
        apply_fn=nnmodel.apply, 
        params=model_state_d['params'],
        batch_stats=model_state_d['batch_stats']
        )