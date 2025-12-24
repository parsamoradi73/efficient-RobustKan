for lr in 1e-2 1e-3 1e-4
do
  for wd in 0 1e-5 1e-4 1e-3 
  do
    echo "Running MLP with lr=${lr}, weight_decay=${wd}"
    python mnist_lightning.py \
      --model mlp \
      --learning-rate $lr \
      --weight-decay $wd \
      --epochs 40 \
      --batch-size 256 \
      --hidden-sizes 1000
  done
done
