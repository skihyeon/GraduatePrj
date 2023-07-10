# GraduatePrj
## 2023 Graduate Project, Using EnlightenGAN, Integrating Self-Attention from SAGAN into EnlightenGAN

### Notion Link
[[Development Note during Prjoect]](https://www.notion.so/Development-note-fa16d62bc29443dabff7f0895537c4e8?pvs=4)  

[[Overview of GraduatePrj]](https://www.notion.so/0e6cedfdda5d4215a2cc6bbe0b77b469?pvs=4)

=======
## Base Paper
### EnlightenGAN: Deep Light Enhancement without Paired Supervision
[Yifan Jiang](https://yifanjiang19.github.io/), Xinyu Gong, Ding Liu, Yu Cheng, Chen Fang, Xiaohui Shen, Jianchao Yang, Pan Zhou, Zhangyang Wang

[[Paper]](https://arxiv.org/abs/1906.06972) [[Supplementary Materials]](https://yifanjiang.net/files/EnlightenGAN_Supplementary.pdf)

=======
## Limits of the Paper
![Limits of the paper](assets/limit.png)

## EnlightenGAN's Generator Architecture
![EnlightenGAN's Generator architecture](assets/original_g.png)  

## Modified Generator Architecture
![Proposed Generator architecture](assets/new_g.png)  
Add Self-Attention Block
form Self-Attention Generative Adversarial Network, Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena

## Used Self-Attention Block Architecture
![Self Attention Block architecture](assets/attn_block.png)

![Self Attention train result](assets/attn_result.png)

## Modified model result
![Proposed model result](assets/EG_result.png)
![](assets/EG_result2.png)

## Improvement
![PSNR/SSIM improvement](assets/PSNR.png)