## VAE  

Command line examples

### Train VAE  

```
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_vae.py --epochs 400 --batch_size 2048 --dataset data/mocap/
```


### Compute LikelihoodRegret

```
$ CUBLAS_WORKSPACE_CONFIG=:4096:8 python compute_LR.py --model exps/exp_2024-10-27-20-03-23/models/model.pt --test-data data/mocap_ood/sitdownchair/reference_motion.pkl
```
