from . import data, infer, net, pipelines, simulation, train
import jax.numpy as jnp

class Config(object):
    
    def __init__(self, data_cfgs, infer_cfgs, train_cfgs):        
        self.data_config = data.DataConfig(**data_cfgs)
        self.infer_config = infer.InferConfig(**infer_cfgs)
        self.train_config = train.TrainConfig(**train_cfgs)
    
    @classmethod 
    def _create(cls):
        # Default configuration
        data.data_config = data.DataConfig()
        infer.infer_config = infer.InferConfig()
        train.train_config = train.TrainConfig()
        simulation.sim_config = simulation.SimulationConfig()
        pipelines.pipeline_config = pipelines.PipelineConfig()
        
    @classmethod
    def configure_LARG_phantom_clinical(cls):
        cls._create()

        # Default for water solution, in case no mapping:
        data.data_config.T1a_ms = 2800
        data.data_config.T2a_ms = 1200
        
        # ==> The "b" pool is L-arginine:
        data.data_config.T2b_ms = 40
        data.data_config.wb_ppm = 3
        
        H2O_mM = 110e3
        Larg_spins = 3 
        max_larg_mM = 150
        max_larg_spin_frac = max_larg_mM * Larg_spins / H2O_mM
        infer.infer_config.fb_scale_fact = max_larg_spin_frac
        infer.infer_config.kb_scale_fact = 1000
                
        data.data_config.B0_base = 3     
        # simulation - keep defaults which are the pulsed thing, following NBMF work        
        
        # ==> Single-slice tiny data, let's converge well
        pipelines.pipeline_config.cest_patience = 100 
        train.train_config.auto_reduce_batch = False 
        
    @classmethod
    def configure_LARG_phantom_preclinical(cls):
        cls.configure_LARG_phantom_clinical()
        
        # ==> Preclinical:
        data.data_config.B0_base = 7        
        simulation.sim_config.DO_SL=False               
        # -- readout: single flip 2D scan??
        simulation.sim_config.num_flip_pulses = 1 
        simulation.sim_config.flip_angle = 90 * jnp.pi / 180  
        # -- saturation: CW (emulate as a single long pulse)
        simulation.sim_config.num_pulses = 1 
        
        infer.infer_config.kb_min = 0
        
        train.train_config.std_up_fact = 0.1
        train.train_config.sim_seq_mode = 'sequential'        
        train.train_config.reglosstype = 'L2'
        
    @classmethod
    def configure_LARG_phantom_preclinical_9T(cls):
        cls.configure_LARG_phantom_preclinical()
        data.data_config.B0_base = 9.4
        infer.infer_config.kb_scale_fact = 1500  # stronger exchange at higher pH
        data.data_config.T1a_ms = 3000
        data.data_config.T2a_ms = 750   # (!) literature: lower at high fields and with the L-arg doping      
        
    @classmethod
    def configure_NOE_preclinical(cls):
        cls._create()
        data.SlicesFeed.norm_type = 'l2'
        
        # ==> The "b" pool is NOE:
        data.data_config.T2b_ms = 5
        data.data_config.wb_ppm = -3.5
        
        infer.infer_config.fb_scale_fact = 0.02
        infer.infer_config.kb_scale_fact = 150        
        
        # ==> Preclinical:
        data.data_config.B0_base = 7        
        simulation.sim_config.DO_SL=False               
        # -- readout --         
        simulation.sim_config.num_flip_pulses = 1 
        simulation.sim_config.flip_angle = 90 * jnp.pi / 180
        # -- saturation
        # CW saturation (emulate as a single long pulse)
        simulation.sim_config.t_pulse = 2.5
        simulation.sim_config.t_delay = 0.01
        simulation.sim_config.num_pulses = 1        

        # ==> Single-slice
        # small data, let's fit the hell out of it
        train.train_config.auto_reduce_batch = False                
        pipelines.pipeline_config.mt_patience = 1000    
        pipelines.pipeline_config.cest_patience = 1000
        pipelines.pipeline_config.mt_lr = pipelines.pipeline_config.cest_lr = 0.01        
        pipelines.pipeline_config.mt_sim_mode = 'expm_bmmat'
        train.train_config.sim_seq_mode = 'sequential' # !! 06-07-2025
        train.train_config.std_up_fact = 0.1
        train.train_config.sigmoid_shrink = 10  
        

    @classmethod
    def configure_mt_amide_preclinical(cls):
        cls._create()
                
        # ==> The "b" pool is AMIDE:
        infer.infer_config.kb_scale_fact = 600
        infer.infer_config.fb_scale_fact = 6e-3
        
        infer.infer_config.kc_scale_fact = 70
        infer.infer_config.fc_scale_fact = 0.3      
        
        # ==> Preclinical:
        data.data_config.B0_base = 7        
        simulation.sim_config.DO_SL=False               
        # -- readout -- 
        # (equivalent to any readout bringing Mz to zero)
        simulation.sim_config.num_flip_pulses = 1 
        simulation.sim_config.flip_angle = 90 * jnp.pi / 180
        # -- saturation
        # CW saturation (emulate as a single long pulse)
        simulation.sim_config.t_pulse = 2.5
        simulation.sim_config.t_delay = 0.01
        simulation.sim_config.num_pulses = 1        

        # ==> Single-slice (small data, let's verify convergence)
        train.train_config.auto_reduce_batch = False                
        pipelines.pipeline_config.mt_lr = pipelines.pipeline_config.cest_lr = 0.01        
        pipelines.pipeline_config.mt_sim_mode = 'expm_bmmat'        
        train.train_config.sigmoid_shrink = 10  
        train.train_config.std_up_fact = 0.1
        train.train_config.sim_seq_mode = 'sequential'
        pipelines.pipeline_config.mt_steps = pipelines.pipeline_config.mt_patience = 1000
        pipelines.pipeline_config.cest_steps = pipelines.pipeline_config.cest_patience = 1000                
                
    @classmethod
    def configure_AMIDE_clinical_whole_brain(cls):
        cls._create()
        # ==> The "b" pool is AMIDE:
        infer.infer_config.kb_scale_fact = 500
        infer.infer_config.fb_scale_fact = 1e-2
        
        infer.infer_config.kc_scale_fact = 70
        infer.infer_config.fc_scale_fact = 0.3
        infer.infer_config.predict_k_k = False  
        
        train.train_config.use_shuffled_sampler = True     
        train.train_config.sim_seq_mode = 'sequential' # ony impacts CEST, not isar2-simulated MT
        # !! L2 more standard for VAEs and appropriate for penalizing points sampled to be far-away
        train.train_config.reglosstype = 'L2'    
        train.train_config.std_up_fact = 0.1   
        train.train_config.weight_decay = 1e-2 

        pipelines.pipeline_config.mt_train_slw = 10        
        pipelines.pipeline_config.mt_steps = 5000            
        pipelines.pipeline_config.cest_steps = 5000  # *16  # 6000
        pipelines.pipeline_config.mt_patience = 200
        pipelines.pipeline_config.cest_patience = 200 # 20   
        pipelines.pipeline_config.add_noise_to_signal = 0.01 