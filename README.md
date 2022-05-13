# No More Update!!!
For the reason that this work leaking of flexibility has some trouble to expand other experiments, I will not update this repository any more. Please move to my other repository [__HEA4ORR__](https://github.com/Wilmido/HEA4ORR) for my latest work.

## Mining High Entropy Alloys as Electrocatalyst for Oxygen Reduction Reaction by Machine Learning
This is repository for high entropy alloys(HEAs) experiments of "<b>【学科交叉项目】神经网络机器学习辅助筛选高熵合金催化剂(\[Interdisciplinary Project\] Neural network machine learning assisted screening of high entropy alloy catalysts)</b>" in "Innovation and Entrepreneurship Project for College Students"(2021), HX2021037

This work is based on [neural-network-design-of-HEA](https://github.com/jol-jol/neural-network-design-of-HEA).
In this repository, I propose a regression model to predict the OH* adsorption energy of HEAs, and a WGAN-GP(Wasserstein GAN using gradient penalty) to generate HEA compositions. Moreover, The regression model is able to tackle with any number of atoms and invariant to input permutaion beacuse I use the symmetric function  __average__ as pooling operation in the model, which greatly helps the model performance in real experiments. 


## Dataset
<b>Because of the ownership of the dataset, this repository doesn't provide HEAs dataset!</b> Therefore, you have to collect your own data! 

The data structure is shown below.

||Atom|Ru|Rh|Pd|Ir|PT|
|--|--|--|--|--|--|--|
|A|Period|5|5|5|6|6|
||Group|8|9|10|9|10|
|B|Radius|1.338|1.345|1.375|1.357|1.387|
|C|CN|
|D|AtSite|
|E|Pauling Negativity|2.20|2.28|2.20|2.20|2.28|
||VEC|8|9|10|9|10|
|F|M|101.07|102.906|106.42|192.2|195.08|
||Atomic Number|44|45|46|77|78|

where CN is coordination number, AtSite is active sites, and M is molar mass. You have to follow the <a href="HEA_selection\data\coord_nums.csv">coord_numbers</a> to fill in the blanks. The rest is filled by 0.

I build up my dataset based on [neural-network-design-of-HEA](https://github.com/jol-jol/neural-network-design-of-HEA), you can refer this repository for more infomation.


## Analysis
It's worth noting that HEAs datasets are a set of atoms and invariant to orders of atoms which require the model we proposed has certain symmetrizations in the net computation[<sup>*</sup>](#refer-anchor-3).


After build up the dataset, you should use Pearson correlation coefficient to drop out highly related features to reduce copmutaion cost, run following code:
```
cd utils
python PearsonSelection.py
```

The result can be seen below:

 <img src="https://user-images.githubusercontent.com/71449089/163708482-4db16267-8b19-4f9e-a3c2-e3d526ae2dfa.png" width = "600" height = "450" alt="Pearson_value" />


The left features are descriptors we deisred, which are denoted as 'A, B, C, D, E, F' in above table.

## Dependecies
The prominent packages are:
* numpy
* pandas
* matplotlib
* scikit-learn
* pytorch 1.8.1

To install all the dependencies quickly and easily, you should use __pip__ install `requirements.txt`
```python
pip install -r requirements.txt
```


## Get Started
To train the model, you can simply use the following command, and you will get a checkpoint:
```
# training a model for downstream tasks
python K_fold.py
```

Obtaining the plot of __MAE__ and __RMSE__ compared with DFT-calculated adsorption energy

```
# training a model for downstream tasks, you need to update the checkpoint path first! 
python main.py
```
![6_500_plot](https://user-images.githubusercontent.com/71449089/163707875-e0862e04-4405-4b16-805e-d58973e49797.svg)


*Pretrained Models*
 ---
You can also just simply use the checkpoint I have provided in <a href="HEA_selection/checkpoint">checkpoint/6_700epochs_5_model.pth</a>.


## T-SNE
Visualize the data, and the features processed by the model. 
```
python t_SNE.py
```
 <img src="https://user-images.githubusercontent.com/71449089/166883621-ab8377d8-6af8-4404-988a-89fe9a79b2b7.svg"  alt="best_data_tsne" />


## Generate HEAs
You can switch the mode to choose whether to train the regression model. The result of loss plot demonstrates that the training process of GAN is not good :(
```
python Joint_training.py
```

![Figure_1](https://user-images.githubusercontent.com/71449089/161955711-a5e78e40-a2df-4e1f-8768-3045bf9f8024.png)


## Reference
<div id="refer-anchor-1"></div>
https://github.com/jol-jol/neural-network-design-of-HEA
<div id="refer-anchor-2"></div>
https://github.com/Zeleni9/pytorch-wgan
<div id="refer-anchor-3"></div>
* https://arxiv.org/pdf/1612.00593.pdf

