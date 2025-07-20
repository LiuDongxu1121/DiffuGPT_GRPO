<div  align="center">
    <h1>diffuGPT-GRPO</h1>
  <p>一个尝试复现的codebase.基于D1实现，所有的个人实现的代码均位于_mini结尾的文件中</p>
</div>



## Environment Setup

To setup the environment, run;
```
conda env create -f env.yml
conda activate grpo
pip install -r requirements.txt
```


## _diffu_-GRPO

再次强调，**mini**结尾的代码才是DiffuGPT-GRPO的相关代码
  ```bash
  cd diffu-GRPO
   #单卡debug
  CUDA_VISIBLE_DEVICES=0, bash run_mini.sh
  #多卡执行
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_mini.sh
  ```
